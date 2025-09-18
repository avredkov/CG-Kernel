from __future__ import annotations

import json
import os
import pickle
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

# Define lightweight model classes compatible with saved state_dicts
from torch import nn


class MLPRegressor(nn.Module):
    """A simple feed-forward network for regression (compatible with training)."""

    def __init__(self, in_features: int, hidden_layers: Optional[List[int]] = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [128, 64]
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.in_features = in_features
        self.hidden_layers = list(hidden_layers)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class MLPClassifier(nn.Module):
    """Feed-forward network for multi-class classification (compatible with training)."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [128, 64]
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=dropout)])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_layers = list(hidden_layers)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


ArrayLike = Union[Sequence[float], np.ndarray, pd.Series, pd.DataFrame]


@dataclass
class FeatureSpec:
    """Specification for a single input feature.

    Attributes:
        name: Canonical feature name as used during training.
        dtype: One of {"float", "int", "categorical"}.
        distribution: One of {"linear", "log"} to indicate expected scale.
        min: Recommended minimum value (for validation/docs).
        max: Recommended maximum value (for validation/docs).
        alias: Optional list of alternative names accepted at inference.
    """

    name: str
    dtype: str
    distribution: str
    min: Optional[float] = None
    max: Optional[float] = None
    alias: Optional[List[str]] = None


class CGKernel:
    """Kernel to load trained models and provide unified prediction APIs.

    Usage overview:
        kernel = CGKernel(config_dir="./kernel")
        y = kernel.predict_property("rms", params_dict)
        cls_idx, cls_proba = kernel.predict_morphology_class(params_dict)
        stab_idx, stab_proba = kernel.predict_stability(params_dict)

    The kernel expects in `config_dir`:
      - models/*.pt and corresponding *.preprocessor.pkl files
      - cgkernel_config.json describing feature schema and model mapping
      - morphologies/*.png for show_moprhology_classes()
    """

    def __init__(self, config_dir: Union[str, os.PathLike]) -> None:
        self.config_dir = Path(config_dir)
        self.models_dir = self.config_dir / "models"
        self.morphologies_dir = self.config_dir / "morphologies"
        self.config_path = self.config_dir / "cgkernel_config.json"
        self.logger = logging.getLogger(__name__)

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        # Load configuration schema
        self.config: Dict[str, Any] = self._load_json(self.config_path)
        # Parse feature specs
        self.feature_specs: List[FeatureSpec] = [
            FeatureSpec(**fs) for fs in self.config.get("features", [])
        ]
        # Map aliases to canonical names
        self.name_alias_map: Dict[str, str] = {}
        for fs in self.feature_specs:
            self.name_alias_map[fs.name.lower()] = fs.name
            for a in (fs.alias or []):
                self.name_alias_map[a.lower()] = fs.name

        # Store training-time preprocessing info snapshot for documentation
        self.numeric_columns: List[str] = self.config.get("numeric_columns", [])
        self.integer_columns: List[str] = self.config.get("integer_columns", [])
        self.log_columns: List[str] = self.config.get("log_columns", [])
        self.categorical_column: str = self.config.get("categorical_column", "Nnucl")
        # Optional mapping from current canonical feature names to training-time names
        self.training_feature_map: Dict[str, str] = {
            str(k): str(v) for k, v in (self.config.get("training_feature_map", {}) or {}).items()
        }

        # Registry for loaded models
        self.regressors: Dict[str, Dict[str, Any]] = {}
        self.classifiers: Dict[str, Dict[str, Any]] = {}

        # Ensure kernel-local packages are importable for unpickling (self-contained)
        import sys
        sys.path.insert(0, str(self.config_dir))
        # Optionally add legacy morphology package dir for unpickling compatibility
        morph_pkg = self.config.get("morphology_package_dir")
        if morph_pkg:
            legacy_dir = self.config_dir / morph_pkg
            if legacy_dir.exists():
                sys.path.insert(0, str(legacy_dir))

        # If ranges_dataset_csv provided, compute ranges to augment feature specs
        self._maybe_compute_feature_ranges()
        # Pre-compute GAN min-max stats from dataset for [0,1] normalization
        self._compute_gan_minmax_stats()

        # Discover and load models according to mapping
        self._load_all_models()

        # Optionally load GAN generator for morphology image synthesis
        self.generator: Optional[Dict[str, Any]] = None
        self._load_generator_if_present()

        # Verbose report
        self._print_loaded_summary()

    # -------------------------- Public API --------------------------
    def describe(self) -> None:
        """Print available predictions and input feature ranges/types."""
        print("CGKernel capabilities:")
        prop_labels = self.config.get("display_property_labels", {}) or {}
        cls_labels = self.config.get("display_classifier_labels", {}) or {}
        def labelize(keys: List[str], mapping: Dict[str, str]) -> List[str]:
            return [mapping.get(k, k.replace("_", " ").title()) for k in keys]
        print("- Continuous properties prediction:", ", ".join(labelize(sorted(self.regressors.keys()), prop_labels)) or "(none)")
        print("- Classification tasks:", ", ".join(labelize(sorted(self.classifiers.keys()), cls_labels)) or "(none)")
        print("- Morphology generation tasks")
        print("Input features (growth conditions) (name | dtype | distribution | range):")
        feat_display_labels: Dict[str, str] = self.config.get("display_labels", {}) or {}
        for fs in self.feature_specs:
            rng = ""
            if fs.min is not None or fs.max is not None:
                rng = f" [{fs.min if fs.min is not None else '-inf'}, {fs.max if fs.max is not None else '+inf'}]"
            disp_name = feat_display_labels.get(fs.name, fs.name)
            print(f"  - {disp_name} | {fs.dtype} | {fs.distribution}{rng}")

    def help(self) -> None:
        """Print brief usage instructions for the kernel API."""
        print("Use predict_property(name, inputs) for: ", ", ".join(self.regressors.keys()))
        print("Use predict_morphology_class(inputs) for morphology cluster classification.")
        print("Use predict_stability(inputs) for stability classification.")
        print("Use generate_morphology(inputs) for generation of surface morphology.\n")
        print("Inputs can be a dict for one sample or a list/DataFrame for batch.")
        print("Call describe() to see expected features and ranges.")

    def predict_property(
        self,
        property_name: str,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        return_linear_units: bool = True,
    ) -> np.ndarray:
        """Predict a continuous property using a trained regressor.

        Args:
            property_name: key of the property, e.g. "growth_rate", "rms".
            inputs: single sample dict or batch (list of dict or DataFrame).
            return_linear_units: if the model was trained on log target, convert back to linear.
        Returns:
            np.ndarray of shape (N,).
        """
        key = self._canonical_property_key(property_name)
        if key not in self.regressors:
            raise KeyError(f"Unknown property '{property_name}'. Known: {sorted(self.regressors.keys())}")

        entry = self.regressors[key]
        model: MLPRegressor = entry["model"]
        pre = entry["preprocessor"]
        device = entry["device"]
        log_target = bool(getattr(pre.config, "log_target", False))

        df = self._inputs_to_dataframe(inputs)
        # Enforce trained range guard (considers configured min/max and log-columns)
        if not self._inputs_within_trained_range(df):
            return "Parameter set is out of trained range"  # type: ignore[return-value]
        # Apply mapping to training-time column names for preprocessor
        df_tr = df.rename(columns=self.training_feature_map) if self.training_feature_map else df.copy()
        # Attach a dummy target to satisfy preprocessor validation
        df_tr[pre.config.target_column] = 1.0
        X, _, _, _ = pre.transform(df_tr)
        with torch.no_grad():
            preds = model(torch.from_numpy(X).to(device)).cpu().numpy().reshape(-1)
        if log_target and return_linear_units:
            preds = (10.0 ** preds).astype(np.float32)
        return preds

    def predict_morphology_class(
        self,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        return_proba: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict morphology cluster label.

        Returns (predicted_class_indices, probabilities_or_None).
        """
        key = self._canonical_classifier_key("morphology")
        entry = self.classifiers.get(key)
        if entry is None:
            raise RuntimeError("Morphology classifier not loaded.")

        model: MLPClassifier = entry["model"]
        pre = entry["preprocessor"]
        device = entry["device"]

        df = self._inputs_to_dataframe(inputs)
        if not self._inputs_within_trained_range(df):
            return "Parameter set is out of trained range", None  # type: ignore[return-value]
        df_tr = df.rename(columns=self.training_feature_map) if self.training_feature_map else df.copy()
        # Attach a dummy target to satisfy preprocessor validation
        df_tr[pre.config.target_column] = int(0)
        X, _, _, _ = pre.transform(df_tr)
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device))
            pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
            proba = torch.softmax(logits, dim=1).cpu().numpy() if return_proba else None
        return pred_idx, proba

    def predict_stability(
        self,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        return_proba: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict stability class (binary/multi) using classifier configured as 'stability'."""
        key = self._canonical_classifier_key("stability")
        entry = self.classifiers.get(key)
        if entry is None:
            raise RuntimeError("Stability classifier not loaded.")

        model: MLPClassifier = entry["model"]
        pre = entry["preprocessor"]
        device = entry["device"]

        df = self._inputs_to_dataframe(inputs)
        if not self._inputs_within_trained_range(df):
            return "Parameter set is out of trained range", None  # type: ignore[return-value]
        df_tr = df.rename(columns=self.training_feature_map) if self.training_feature_map else df.copy()
        df_tr[pre.config.target_column] = int(0)
        X, _, _, _ = pre.transform(df_tr)
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device))
            pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
            proba = torch.softmax(logits, dim=1).cpu().numpy() if return_proba else None
        return pred_idx, proba

    def show_morphology_classes(self, cols: int = 4) -> None:  # keeping requested name
        """Display morphology class PNGs as a panel."""
        import matplotlib.pyplot as plt

        if not self.morphologies_dir.exists():
            raise FileNotFoundError(f"Morphology images directory not found: {self.morphologies_dir}")
        # Collect numerically named PNGs and optional Colormap.png (shown as "Height")
        numeric_pairs: List[Tuple[int, Path]] = []
        colormap_path: Optional[Path] = None
        for p in self.morphologies_dir.glob("*.png"):
            stem = p.stem
            if str(stem).lower() == "colormap":
                colormap_path = p
                continue
            try:
                if str(stem).isdigit():
                    numeric_pairs.append((int(stem), p))
            except Exception:
                continue
        numeric_pairs.sort(key=lambda t: t[0])
        entries: List[Tuple[Path, str]] = [(p, str(num)) for num, p in numeric_pairs]
        if colormap_path is not None:
            entries.append((colormap_path, "Height"))
        if not entries:
            print("No PNGs found in:", self.morphologies_dir)
            return
        n = len(entries)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, (path, title) in zip(axes, entries):
            img = plt.imread(str(path))
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
        # Hide any extra axes
        for ax in axes[n:]:
            ax.axis("off")
        fig.tight_layout()
        plt.show()



    # -------------------------- Internals --------------------------
    def _inputs_to_dataframe(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
        """Normalize various input forms into a DataFrame with canonical feature names."""
        if isinstance(inputs, pd.DataFrame):
            df = inputs.copy()
        elif isinstance(inputs, dict):
            df = pd.DataFrame([inputs])
        elif isinstance(inputs, list):
            df = pd.DataFrame(inputs)
        else:
            raise TypeError("inputs must be dict, list[dict], or DataFrame")

        # Rename columns using alias mapping to canonical names
        rename_map: Dict[str, str] = {}
        for col in df.columns:
            canon = self.name_alias_map.get(str(col).lower())
            if canon is not None and canon != col:
                rename_map[col] = canon
        if rename_map:
            df = df.rename(columns=rename_map)

        # Ensure required columns exist; if missing, try defaults from config
        required = set(self.numeric_columns + self.integer_columns + [self.categorical_column])
        missing = required.difference(df.columns)
        if missing:
            defaults: Dict[str, Any] = self.config.get("defaults", {})
            for m in list(missing):
                if m in defaults:
                    df[m] = defaults[m]
            missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"Missing required input columns: {sorted(missing)}")

        # Type coercion according to feature specs
        for fs in self.feature_specs:
            if fs.name in df.columns:
                if fs.dtype == "float":
                    df[fs.name] = pd.to_numeric(df[fs.name], errors="coerce")
                elif fs.dtype == "int":
                    # Round to nearest integer before casting to nullable Int64 to avoid
                    # pandas safe-cast errors when grid values are non-integer floats.
                    s_int = pd.to_numeric(df[fs.name], errors="coerce").round()
                    df[fs.name] = s_int.astype("Int64").astype(float)
                elif fs.dtype == "categorical":
                    df[fs.name] = pd.to_numeric(df[fs.name], errors="coerce").astype(int)

        # Basic range validation (warn only)
        for fs in self.feature_specs:
            if fs.name in df.columns and (fs.min is not None or fs.max is not None):
                vals = pd.to_numeric(df[fs.name], errors="coerce")
                out_of_range = ((fs.min is not None) & (vals < fs.min)) | (
                    (fs.max is not None) & (vals > fs.max)
                )
                if bool(np.any(out_of_range.fillna(False))):
                    self.logger.warning(f"Values for {fs.name} are outside recommended range [{fs.min}, {fs.max}]. This might indicate an issue with input data or model training.")
        return df

    def _canonical_property_key(self, name: str) -> str:
        name = name.strip().lower().replace(" ", "_")
        mapping: Dict[str, str] = self.config.get("regressor_keys", {})
        return mapping.get(name, name)

    def _canonical_classifier_key(self, name: str) -> str:
        name = name.strip().lower().replace(" ", "_")
        mapping: Dict[str, str] = self.config.get("classifier_keys", {})
        return mapping.get(name, name)

    def _print_loaded_summary(self) -> None:
        self.logger.info("Loaded models from: %s", self.models_dir)
        for k, ent in sorted(self.regressors.items()):
            pre = ent["preprocessor"]
            log_tgt = getattr(pre.config, "log_target", False)
            self.logger.info(
                "  - Regressor '%s': in_features=%s, hidden=%s, dropout=%s, log_target=%s",
                k,
                ent['model'].in_features,
                ent['model'].hidden_layers,
                ent['model'].dropout,
                log_tgt,
            )
        for k, ent in sorted(self.classifiers.items()):
            pre = ent["preprocessor"]
            self.logger.info(
                "  - Classifier '%s': in_features=%s, num_classes=%s, hidden=%s, dropout=%s",
                k,
                ent['model'].in_features,
                ent['model'].num_classes,
                ent['model'].hidden_layers,
                ent['model'].dropout,
            )
        # Print expected inputs
        self.describe()

    def _load_all_models(self) -> None:
        """Load regressors and classifiers based on configuration mapping.

        Expected config sections:
          - models.regressors: { key: { pt: "file.pt", pkl: "file.preprocessor.pkl" } }
          - models.classifiers: { key: { pt: "file.pt", pkl: "file.preprocessor.pkl", num_classes: int? } }
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models_cfg = self.config.get("models", {})

        # Load regressors
        for key, spec in models_cfg.get("regressors", {}).items():
            pt_path = self.models_dir / spec["pt"]
            pkl_path = self.models_dir / spec["pkl"]
            pre = self._load_pickle(pkl_path)
            state = torch.load(pt_path, map_location=device)
            sizes = self._extract_linear_sizes(state)
            model = self._build_exact_mlp_from_sizes(sizes, dropout=float(spec.get("dropout", 0.1)))
            model.load_state_dict(state, strict=True)
            model.eval().to(device)
            # Expose attributes for summary
            model.in_features = int(sizes[0][0]) if sizes else 0  # type: ignore[attr-defined]
            model.hidden_layers = [int(s[1]) for s in sizes[:-1]]  # type: ignore[attr-defined]
            model.dropout = float(spec.get("dropout", 0.1))  # type: ignore[attr-defined]
            self.regressors[key] = {"model": model, "preprocessor": pre, "device": device}

        # Load classifiers
        for key, spec in models_cfg.get("classifiers", {}).items():
            pt_path = self.models_dir / spec["pt"]
            pkl_path = self.models_dir / spec["pkl"]
            pre = self._load_pickle(pkl_path)
            state = torch.load(pt_path, map_location=device)
            sizes = self._extract_linear_sizes(state)
            model = self._build_exact_mlp_from_sizes(sizes, classifier=True, dropout=float(spec.get("dropout", 0.1)))
            model.load_state_dict(state, strict=True)
            model.eval().to(device)
            # Expose attributes for summary
            model.in_features = int(sizes[0][0]) if sizes else 0  # type: ignore[attr-defined]
            model.num_classes = int(sizes[-1][1]) if sizes else 0  # type: ignore[attr-defined]
            model.hidden_layers = [int(s[1]) for s in sizes[:-1]]  # type: ignore[attr-defined]
            model.dropout = float(spec.get("dropout", 0.1))  # type: ignore[attr-defined]
            self.classifiers[key] = {"model": model, "preprocessor": pre, "device": device}

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_pickle(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor pickle not found: {path}")
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except ModuleNotFoundError as e:
            self.logger.warning("Module not found during unpickle (%s). Retrying with aliasing.", e)
            import io

            class _AliasUnpickler(pickle.Unpickler):
                MODULE_ALIASES = {
                    # legacy module path -> current path mappings can be added here if needed
                }

                def find_class(self, module: str, name: str):  # type: ignore[override]
                    module = self.MODULE_ALIASES.get(module, module)
                    return super().find_class(module, name)

            with open(path, "rb") as f:
                data = f.read()
            return _AliasUnpickler(io.BytesIO(data)).load()

    # -------- helpers: infer architecture and ranges --------
    def _infer_mlp_hidden_from_state(
        self,
        state: Dict[str, Any],
        in_features: int,
        is_classifier: bool,
        fallback_hidden: Optional[List[int]] = None,
        fallback_dropout: float = 0.1,
        final_out: Optional[int] = None,
    ) -> Tuple[List[int], float]:
        """Infer hidden layer sizes from a saved state_dict of Sequential MLP.

        Assumes keys like 'net.0.weight', 'net.3.weight', ... for Linear layers.
        """
        # Collect linear layer weights in numerical order by layer index (net.{idx}.weight)
        items: List[Tuple[int, torch.Tensor]] = []
        for k, v in state.items():
            if not k.endswith(".weight"):
                continue
            if not k.startswith("net."):
                continue
            if not isinstance(v, torch.Tensor) or v.ndim != 2:
                continue
            try:
                idx = int(k.split(".")[1])
            except Exception:
                continue
            items.append((idx, v))
        items.sort(key=lambda t: t[0])
        sizes: List[Tuple[int, int]] = [(int(w.shape[1]), int(w.shape[0])) for _, w in items]
        # Expect at least one linear
        hidden: List[int] = []
        if sizes:
            # sizes includes all Linear layers including final.
            # Infer dropout from presence of Dropout modules is non-trivial from state_dict; use fallback.
            for i, (fin, fout) in enumerate(sizes):
                # First linear should match provided in_features; tolerate mismatch
                is_last = (i == len(sizes) - 1)
                if not is_last:
                    hidden.append(fout)
            # If classifier and final_out provided, ensure last out matches
            # (We don't enforce; just informational.)
        else:
            hidden = list(fallback_hidden or [128, 64])
        return hidden, float(fallback_dropout)

    def _infer_num_classes_from_state(self, state: Dict[str, Any]) -> int:
        # Find last linear weight and return its out_features
        linear_weights = [(k, v) for k, v in state.items() if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2]
        if not linear_weights:
            return 2
        last_key, last_w = sorted(linear_weights, key=lambda kv: kv[0])[-1]
        return int(last_w.shape[0])

    def _maybe_compute_feature_ranges(self) -> None:
        dataset = self.config.get("ranges_dataset_csv")
        if not dataset:
            return
        path = Path(dataset)
        if not path.is_file():
            return
        try:
            df = pd.read_csv(path)
        except Exception:
            return
        # Compute 1st and 99th percentiles for numeric/int features
        spec_by_name = {fs.name: fs for fs in self.feature_specs}
        cols = self.numeric_columns + self.integer_columns
        for col in cols:
            if col in df.columns and col in spec_by_name:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                q1, q99 = np.percentile(vals.values, [1.0, 99.0])
                spec = spec_by_name[col]
                spec.min = float(q1)
                spec.max = float(q99)

    def _extract_linear_sizes(self, state: Dict[str, Any]) -> List[Tuple[int, int]]:
        indices: List[int] = []
        for k in state.keys():
            if k.endswith(".weight") and k.startswith("net."):
                try:
                    idx = int(k.split(".")[1])
                except Exception:
                    continue
                indices.append(idx)
        indices = sorted(set(indices))
        sizes: List[Tuple[int, int]] = []
        for idx in indices:
            w = state.get(f"net.{idx}.weight")
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                sizes.append((int(w.shape[1]), int(w.shape[0])))
        return sizes

    def _build_exact_mlp_from_sizes(self, sizes: List[Tuple[int, int]], classifier: bool = False, dropout: float = 0.1) -> nn.Module:
        layers: List[nn.Module] = []
        for i, (in_f, out_f) in enumerate(sizes):
            layers.append(nn.Linear(in_f, out_f))
            if i < len(sizes) - 1:
                layers.extend([nn.ReLU(), nn.Dropout(p=dropout)])
        net = nn.Sequential(*layers)
        class Wrapper(nn.Module):
            def __init__(self, net: nn.Sequential) -> None:
                super().__init__()
                self.net = net
            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                return self.net(x)
        return Wrapper(net)

    def _weights_candidates_for_generator(self) -> List[Path]:
        gen_cfg = self.config.get("generator", {})
        candidates: List[Path] = []
        # explicit file
        g_path = gen_cfg.get("pth") or gen_cfg.get("pt")
        if isinstance(g_path, str) and g_path:
            p = Path(g_path)
            if p.is_absolute():
                if p.exists():
                    candidates.append(p)
            else:
                # try relative to config_dir (handles "./models/..")
                p1 = (self.config_dir / p).resolve()
                if p1.exists():
                    candidates.append(p1)
                # also try inside models_dir if a bare filename or nested under generator
                p2 = (self.models_dir / (p.name if p.name else p)).resolve()
                if p2.exists() and p2 not in candidates:
                    candidates.append(p2)
        # conventional locations
        gen_dir = self.models_dir / "generator"
        if gen_dir.exists():
            try:
                items = sorted(gen_dir.glob("*.pth"), key=lambda t: t.stat().st_mtime, reverse=True)
                candidates.extend([pp for pp in items if pp.exists()])
            except Exception:
                pass
        # checkpoints
        ckpt_latest = self.models_dir / "checkpoints" / "checkpoint_latest.pth"
        if ckpt_latest.exists():
            candidates.append(ckpt_latest)
        # TorchScript preferred
        ts_path = self.models_dir / "generator.ts"
        if ts_path.exists():
            candidates.insert(0, ts_path)
        # de-dup
        uniq: List[Path] = []
        seen = set()
        for p in candidates:
            key = str(p.resolve())
            if key not in seen and p.exists():
                uniq.append(p)
                seen.add(key)
        if not uniq:
            self.logger.info("No generator weight candidates found. Checked config 'generator.pth', 'models/generator/*.pth', 'models/checkpoints/checkpoint_latest.pth', and 'models/generator.ts'.")
        else:
            self.logger.debug("Generator weight candidates: %s", ", ".join(str(x) for x in uniq))
        return uniq

    def _strip_module_prefix(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if any(str(k).startswith("module.") for k in state.keys()):
            return {str(k).replace("module.", "", 1): v for k, v in state.items()}
        return state

    def _load_generator_state(self, p: Path) -> Dict[str, Any]:
        sd = torch.load(str(p), map_location="cpu")
        if isinstance(sd, dict) and "generator_state" in sd:
            st = sd["generator_state"]
        elif isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            st = sd["state_dict"]
        else:
            st = sd
        st = self._strip_module_prefix(st)
        # quick validation: require at least one first deconv/conv transpose weight
        if not any(str(k).endswith("deconv1.weight") or str(k).endswith("layer1.0.weight") for k in st.keys()):
            raise RuntimeError("Provided generator weights do not look like a Generator state_dict")
        return st

    def _infer_generator_hparams(self, state: Dict[str, Any]) -> Dict[str, Any]:
        keys = list(state.keys())
        def find_key(suf: str) -> Optional[str]:
            for k in keys:
                if str(k).endswith(suf):
                    return str(k)
            return None
        k_l1 = find_key("deconv1.weight") or find_key("layer1.0.weight")
        if k_l1 is None:
            raise RuntimeError("Missing first deconv weight in generator state")
        w_l1 = state[k_l1]
        in_ch_l1 = int(w_l1.shape[0])
        init_dim = int(w_l1.shape[1])
        k_smod = find_key("smod1.affine.weight")
        style_in = int(state[k_smod].shape[1]) if k_smod is not None else in_ch_l1
        has_label_proj = any(str(k).endswith("label_proj.0.weight") for k in keys)
        cond_embed_dim_g = 0
        label_input_dim = 0
        if has_label_proj:
            k_lp = find_key("label_proj.0.weight")
            if k_lp is not None:
                w_lp = state[k_lp]
                cond_embed_dim_g = int(w_lp.shape[0])
                label_input_dim = int(w_lp.shape[1])
        # concat vs style conditioning
        if has_label_proj and k_smod is not None:
            diff = style_in - in_ch_l1
            if diff == cond_embed_dim_g:
                g_concat = False
                latent_size = in_ch_l1
            elif diff == 0:
                g_concat = True
                latent_size = in_ch_l1 - cond_embed_dim_g
            else:
                g_concat = False
                latent_size = style_in
        else:
            g_concat = False
            latent_size = in_ch_l1
        k_final = find_key("final.0.weight")
        if k_final is None:
            raise RuntimeError("Missing final conv weight in generator state")
        w_final = state[k_final]
        n_channel = int(w_final.shape[1])
        regime = "C-GAN" if has_label_proj else "GAN"
        return dict(
            regime=regime,
            latent_size=int(latent_size),
            cond_embed_dim_g=int(cond_embed_dim_g),
            label_input_dim=int(label_input_dim),
            n_channel=int(n_channel),
            init_dim=int(init_dim),
            g_concat_cond_to_input=bool(g_concat),
            style_in=int(style_in),
        )

    def _build_label_vector(self, values: Dict[str, Any]) -> np.ndarray:
        # training order: numeric (self.numeric_columns then self.integer_columns), then categorical one-hot for Regime
        parts: List[float] = []
        # numeric + integer scaled to [0,1] or standardized? We don't have training scaler here; use minmax from feature specs if present
        # If no ranges provided, pass raw values (generator expected standardized minmax in training; mismatch may affect quality)
        for name in self.numeric_columns + self.integer_columns:
            v = float(values[name])
            spec = next((fs for fs in self.feature_specs if fs.name == name), None)
            if spec and spec.min is not None and spec.max is not None and spec.max > spec.min:
                x = (v - float(spec.min)) / (float(spec.max) - float(spec.min))
            else:
                x = v
            parts.append(float(x))
        # categorical one-hot
        regime_name = self.categorical_column
        # Assume regimes are 0..K-1 with K from spec.max
        spec_cat = next((fs for fs in self.feature_specs if fs.name == regime_name), None)
        k = int(spec_cat.max) + 1 if (spec_cat and spec_cat.max is not None) else 3
        idx = int(values[regime_name])
        if idx < 0 or idx >= k:
            raise ValueError(f"{regime_name} out of range: {idx} not in [0,{k})")
        one_hot = [0.0] * k
        one_hot[idx] = 1.0
        parts.extend(one_hot)
        return np.array(parts, dtype=np.float32)[None, :]

    def _load_generator_if_present(self) -> None:
        try:
            cands = self._weights_candidates_for_generator()
            if not cands:
                self.generator = None
                return
            p = cands[0]
            # Try TorchScript first regardless of extension
            try:
                gen_ts = torch.jit.load(str(p), map_location="cpu")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                gen_ts.eval().to(device)
                # Try inferring hparams from scripted module's state_dict
                hp_ts: Dict[str, Any] = {}
                try:
                    hp_ts = self._infer_generator_hparams(gen_ts.state_dict())  # type: ignore[arg-type]
                except Exception:
                    hp_ts = {}
                self.generator = {"model": gen_ts, "hparams": hp_ts, "device": device}
                self.logger.info("Loaded TorchScript GAN generator from %s", p)
                return
            except Exception:
                pass
            state = self._load_generator_state(p)
            hp = self._infer_generator_hparams(state)
            # Check for user-provided import path and optional module_path
            gen_cfg = self.config.get("generator", {})
            gen_spec = gen_cfg.get("import")
            module_path = gen_cfg.get("module_path")
            gen = None
            if isinstance(module_path, str) and module_path:
                import sys
                mpath = Path(module_path)
                if not mpath.is_absolute():
                    mpath = (self.config_dir / mpath).resolve()
                sys.path.insert(0, str(mpath))
                self.logger.debug("Added module_path to sys.path: %s", mpath)
            if isinstance(gen_spec, str) and ":" in gen_spec:
                try:
                    mod_name, cls_name = gen_spec.split(":", 1)
                    import importlib
                    mod = importlib.import_module(mod_name)
                    GenCls = getattr(mod, cls_name)
                    try:
                        gen = GenCls(hp)
                    except Exception:
                        try:
                            gen = GenCls(**hp)
                        except Exception as e:
                            self.logger.warning("Failed to instantiate Generator with inferred hparams: %s", e)
                            gen = GenCls()
                except Exception as e:
                    self.logger.warning("Could not import Generator '%s': %s", gen_spec, e)
            if gen is None:
                # Fallback: require TorchScript or import path
                self.logger.warning("No Generator implementation provided. Set generator.import/module_path in config or export TorchScript to models/generator.ts.")
                self.generator = None
                return
            # Load weights if supported
            try:
                gen.load_state_dict(state, strict=False)
            except Exception as e:
                self.logger.warning("Could not load generator state_dict strictly: %s", e)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gen.eval().to(device)
            self.generator = {"model": gen, "hparams": hp, "device": device}
            self.logger.info("Loaded GAN generator from %s", p)
        except Exception as e:
            self.logger.warning("Failed to load GAN generator: %s", e)
            self.generator = None

    def _compute_gan_minmax_stats(self) -> None:
        """Compute per-feature min/max for GAN conditioning from config features.

        Uses FeatureSpec.min/max from cgkernel_config.json. For columns listed in
        self.log_columns, the configured min/max are assumed to be pre-log values
        and are transformed with log10 here to match conditioning (which applies
        log10 to inputs). Falls back to dataset CSV only if config bounds are
        missing.
        """
        self.gan_minmax: Dict[str, Tuple[float, float]] = {}
        cols = self.numeric_columns + self.integer_columns
        # 1) Prefer explicit bounds from config features
        for fs in self.feature_specs:
            if fs.name not in cols:
                continue
            if fs.min is None or fs.max is None or not (fs.max > fs.min):
                continue
            vmin = float(fs.min)
            vmax = float(fs.max)
            if fs.name in (self.log_columns or []):
                # Config values are pre-log; convert to log10 scale for conditioning
                if vmin <= 0.0 or vmax <= 0.0:
                    self.logger.warning("Skipping GAN min/max for '%s' due to non-positive bounds for log10.", fs.name)
                    continue
                vmin = float(np.log10(vmin))
                vmax = float(np.log10(vmax))
            # Guard equal bounds
            if vmax == vmin:
                vmax = vmin + 1.0
            self.gan_minmax[fs.name] = (vmin, vmax)
        # 2) Fallback to dataset CSV if some columns lack bounds in config
        missing = [c for c in cols if c not in self.gan_minmax]
        dataset = self.config.get("ranges_dataset_csv")
        if missing and dataset:
            path = Path(dataset)
            if not path.is_absolute():
                path = (self.config_dir / path).resolve()
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    for name in missing:
                        if name in df.columns:
                            s = pd.to_numeric(df[name], errors="coerce").dropna().astype(float)
                            if len(s) == 0:
                                continue
                            vmin = float(np.min(s.values))
                            vmax = float(np.max(s.values))
                            if name in (self.log_columns or []):
                                if vmin <= 0.0 or vmax <= 0.0:
                                    continue
                                vmin = float(np.log10(vmin))
                                vmax = float(np.log10(vmax))
                            if vmax == vmin:
                                vmax = vmin + 1.0
                            self.gan_minmax[name] = (vmin, vmax)
                except Exception:
                    pass

    def _apply_log_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.log_columns:
            return df
        df2 = df.copy()
        for col in self.log_columns:
            if col in df2.columns:
                vals = pd.to_numeric(df2[col], errors="coerce")
                if bool((vals <= 0).any()):
                    raise ValueError(f"Log-transform requires positive values for column '{col}'.")
                df2[col] = np.log10(vals)
        return df2

    def _inputs_within_trained_range(self, df: pd.DataFrame) -> bool:
        """Return True if all rows are within configured training ranges.

        The check uses `FeatureSpec.min/max` from config. For features listed in
        `self.log_columns`, bounds are interpreted in raw (pre-log) space and
        inputs must be strictly positive. If any sample in the batch violates
        a bound, the whole batch is considered out-of-range.
        """
        if df is None or len(df) == 0:
            return False
        for fs in self.feature_specs:
            name = fs.name
            if name not in df.columns:
                continue
            s = pd.to_numeric(df[name], errors="coerce")
            # NaNs imply invalid inputs
            if bool(s.isna().any()):
                return False
            # Log-space features: must be positive in raw inputs
            if name in (self.log_columns or []):
                if bool((s <= 0).any()):
                    return False
            # Lower bound
            if fs.min is not None:
                if bool((s < float(fs.min)).any()):
                    return False
            # Upper bound
            if fs.max is not None:
                if bool((s > float(fs.max)).any()):
                    return False
            # Categorical guard: ensure within integer category range if provided
            if fs.dtype == "categorical":
                s_int = s.astype(int)
                if fs.min is not None and bool((s_int < int(fs.min)).any()):
                    return False
                if fs.max is not None and bool((s_int > int(fs.max)).any()):
                    return False
        return True

    @torch.inference_mode()
    def generate_morphology(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame], seed: Optional[int] = None, scale_by_peak_to_valley: bool = False, show_3d: bool = False) -> np.ndarray:
        """Generate a surface image using the trained GAN generator.

        Args:
            inputs: single sample dict/list/df with the same feature set used for classifiers/regressors.
            seed: optional random seed for reproducibility.
            scale_by_peak_to_valley: if True, post-scale each image by the predicted peak-to-valley amplitude from the corresponding regressor.
            show_3d: deprecated; no plotting is performed.
        Returns:
            2D array (H, W) float16 for single input, or 3D array (N, H, W) float16 for batch.
        """
        if self.generator is None:
            raise RuntimeError("Generator not loaded. Either place a TorchScript file at models/generator.ts (or point 'generator.pth' to a TorchScript file) OR set 'generator.import' and optional 'generator.module_path' in cgkernel_config.json to import your Generator class and supply a state_dict.")
        gen = self.generator["model"]
        device = self.generator["device"]
        hp = dict(self.generator.get("hparams", {}))
        # Normalize inputs and build conditioning matrix following training order
        df = self._inputs_to_dataframe(inputs)
        batch_size = int(len(df))
        # Build label matrix (N, D) for conditioning
        mat = self._build_label_matrix(df)  # (N, D)
        expected_label_dim = int(hp.get("label_input_dim", mat.shape[1] if mat.shape[1] > 0 else 0))
        # If the generator expects labels (conditional), ensure we have a correctly sized y
        y_vec = None
        if expected_label_dim > 0:
            if mat.shape[1] != expected_label_dim:
                # Attempt to reconcile by rebuilding categorical one-hot to expected cardinality
                num_int_cols: List[str] = self.numeric_columns + self.integer_columns
                n_num = len(num_int_cols)
                exp_cat = max(expected_label_dim - n_num, 0)
                # numeric part, recomputed with log/minmax
                df_proc = self._apply_log_columns(df)
                Xn = df_proc[num_int_cols].astype(float).to_numpy(copy=True) if n_num > 0 else np.zeros((batch_size, 0), dtype=np.float32)
                # categorical one-hot with expected size
                if exp_cat > 0:
                    regime_name = self.categorical_column
                    idx = pd.to_numeric(df_proc[regime_name], errors="coerce").fillna(0).astype(int).to_numpy()
                    idx = np.clip(idx, 0, exp_cat - 1)
                    Xc = np.eye(exp_cat, dtype=np.float32)[idx]
                else:
                    Xc = np.zeros((batch_size, 0), dtype=np.float32)
                mat = np.concatenate([Xn.astype(np.float32), Xc], axis=1)
            # Final guard: pad or trim to expected dimension
            if mat.shape[1] < expected_label_dim:
                pad = np.zeros((batch_size, expected_label_dim - mat.shape[1]), dtype=np.float32)
                mat = np.concatenate([mat, pad], axis=1)
            elif mat.shape[1] > expected_label_dim:
                mat = mat[:, :expected_label_dim]
            y_vec = torch.from_numpy(mat).to(device)
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        z_dim = int(hp.get("latent_size", 128))
        z = torch.randn(batch_size, z_dim, 1, 1, device=device)
        # Call generator respecting whether it expects labels
        if expected_label_dim > 0:
            if y_vec is None:
                raise RuntimeError("Conditional generator requires label vector 'y', but it could not be constructed.")
            out = gen(z, y_vec)  # type: ignore[misc]
        else:
            # Unconditional
            try:
                out = gen(z)  # type: ignore[misc]
            except TypeError:
                # Some modules still accept (z, None)
                out = gen(z, None)  # type: ignore[misc]
        # Postprocess to 2D images (per-sample min-max scaling)
        t = out.detach().float().cpu()
        # Expect (N, C, H, W) or (C, H, W) or (N, H, W)
        if t.ndim == 3:
            # (C, H, W) -> (1, C, H, W)
            t = t.unsqueeze(0)
        if t.ndim == 4 and t.shape[1] > 1:
            # take first channel
            t = t[:, 0:1, :, :]
        if t.ndim == 4:
            # (N, 1, H, W) -> (N, H, W)
            t = t[:, 0, :, :]
        arr = t.numpy()  # (N, H, W) or (1, H, W)
        if arr.ndim == 2:
            arr = arr[None, ...]
        n, h, w = arr.shape
        flat = arr.reshape(n, -1)
        mins = np.nanmin(flat, axis=1)
        maxs = np.nanmax(flat, axis=1)
        denom = (maxs - mins)
        denom[denom <= 0] = 1.0
        # Per-sample min-max to [0, 1]
        scaled = ((arr - mins[:, None, None]) / denom[:, None, None])
        # Optional amplitude scaling using peak_to_valley regressor
        if scale_by_peak_to_valley:
            try:
                amps = self.predict_property("peak_to_valley", df)  # (N,)
                if amps is not None and len(amps) == arr.shape[0]:
                    scaled = scaled * amps.reshape(-1, 1, 1)
            except Exception as e:
                self.logger.warning("peak_to_valley scaling failed; returning unscaled values. Error: %s", e)
        # Return float16 for efficiency
        imgs = np.clip(scaled, 0.0, None).astype(np.float16)
        # Return 2D for single sample, or 3D (N, H, W) for batch
        result = imgs[0] if batch_size == 1 else imgs
        return result

    def _build_label_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Build batched conditioning matrix (N, D) for the GAN.

        Steps:
        - Apply log10 to columns listed in self.log_columns (values must be > 0)
        - Min-max scale numeric+integer columns to [0,1] using self.gan_minmax (config-driven)
        - One-hot encode the categorical 'Regime' with size from FeatureSpec.max+1 when available, else from data
        """
        df_proc = self._apply_log_columns(df)
        parts: List[np.ndarray] = []
        # Numeric + integer
        num_int_cols: List[str] = self.numeric_columns + self.integer_columns
        if num_int_cols:
            arr = df_proc[num_int_cols].astype(float).to_numpy(copy=True)
            for j, name in enumerate(num_int_cols):
                if name in getattr(self, "gan_minmax", {}):
                    vmin, vmax = self.gan_minmax[name]
                    denom = (vmax - vmin) if (vmax > vmin) else 1.0
                    arr[:, j] = (arr[:, j] - vmin) / denom
            parts.append(arr.astype(np.float32))
        # Categorical one-hot
        regime_name = self.categorical_column
        if regime_name in df_proc.columns:
            spec_cat = next((fs for fs in self.feature_specs if fs.name == regime_name), None)
            k = int(spec_cat.max) + 1 if (spec_cat and spec_cat.max is not None) else int(pd.to_numeric(df_proc[regime_name], errors="coerce").fillna(0).astype(int).max()) + 1
            idx = pd.to_numeric(df_proc[regime_name], errors="coerce").fillna(0).astype(int).to_numpy()
            idx = np.clip(idx, 0, max(k - 1, 0))
            one_hot = np.eye(max(k, int(idx.max()) + 1), dtype=np.float32)[idx]
            parts.append(one_hot)
        if not parts:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.concatenate(parts, axis=1)


