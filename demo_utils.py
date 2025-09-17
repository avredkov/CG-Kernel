from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

def _apply_numpy_compatibility_patch():
    """
    Apply compatibility patches for deprecated NumPy attributes.
    
    This function adds missing attributes that older versions of Plotly
    expect to exist in NumPy, preventing AttributeError exceptions.
    """
    # Add np.bool8 as an alias for np.bool_ (for Plotly 5.10 compatibility)
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
        warnings.warn(
            "Added np.bool8 compatibility alias for Plotly 5.10. "
            "Consider upgrading to Plotly >= 5.17 for better NumPy compatibility.",
            DeprecationWarning,
            stacklevel=3
        )
    
    # Add other potentially missing aliases
    if not hasattr(np, 'int0'):
        np.int0 = np.int_
    
    if not hasattr(np, 'uint0'):
        np.uint0 = np.uint


# Apply NumPy compatibility patch after function definition
_apply_numpy_compatibility_patch()
import plotly.express as px
import plotly.graph_objects as go


LOGGER = logging.getLogger(__name__)


@dataclass
class Colormap:
    """Container for a discrete set of colors and an optional Plotly colorscale.

    Attributes:
        discrete: List of hex colors (e.g., ["#112233", ...]) for category coloring.
        scale: Plotly-compatible continuous colorscale as list of (float, color).
    """

    discrete: List[str]
    scale: List[Tuple[float, str]]


def discover_project_root(start: Path | str = ".") -> Path:
    """Return the project root where `cgkernel_config.json` resides.

    Args:
        start: Starting directory.

    Returns:
        Path to project root.
    """
    p = Path(start).resolve()
    for _ in range(6):
        if (p / "cgkernel_config.json").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(start).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataframe_from_config(root: Path, config: Dict[str, Any]) -> pd.DataFrame:
    """Load the dataset CSV specified in config["ranges_dataset_csv"].

    The path may be relative to the project root.
    """
    csv_rel = config.get("ranges_dataset_csv")
    if not csv_rel:
        raise FileNotFoundError("ranges_dataset_csv is not set in cgkernel_config.json")
    csv_path = Path(csv_rel)
    if not csv_path.is_absolute():
        csv_path = (root / csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    # Auto-detect delimiter (handles comma/semicolon, etc.)
    df = pd.read_csv(csv_path, sep=None, engine="python")
    # Normalize header whitespace
    try:
        df.columns = df.columns.str.strip()
    except Exception:
        pass
    # Drop unnamed auto-index columns if present
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # Filter out rows where growth_rate == 0 (if column exists)
    if "growth_rate" in df.columns:
        gr = pd.to_numeric(df["growth_rate"], errors="coerce")
        df = df.loc[gr != 0].copy()
    return df


def percentile_range(series: pd.Series, p1: float = 1.0, p2: float = 99.0) -> Tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float).values
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [p1, p2])
    if not np.isfinite(lo):
        lo = float(np.nanmin(values))
    if not np.isfinite(hi):
        hi = float(np.nanmax(values))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return float(lo), float(hi)


def load_colormap_from_png(path: Path, num_colors: int = 256) -> Colormap:
    """Extract a colorscale and a discrete palette from a horizontal PNG gradient.

    Args:
        path: Path to PNG file.
        num_colors: Number of colors to sample.

    Returns:
        Colormap with discrete colors and a Plotly colorscale.
    """
    try:
        from PIL import Image  # lazy import
    except Exception as exc:
        raise RuntimeError("Pillow is required to read PNG colormap. Install with `pip install pillow`.") from exc

    img = Image.open(str(path)).convert("RGB")
    img = img.resize((num_colors, 1))
    colors = np.array(img)[0]
    hex_list = [f"#{r:02X}{g:02X}{b:02X}" for (r, g, b) in colors]
    scale = [(i / (num_colors - 1), f"rgb({r},{g},{b})") for i, (r, g, b) in enumerate(colors)]
    return Colormap(discrete=hex_list, scale=scale)


def find_colormap_png(root: Path, config: Dict[str, Any]) -> Optional[Path]:
    """Try to find a palette PNG similar to examples.

    Search order:
      1) config["colormap_png"] if present
      2) project root file named "Colormap.png"
      3) any PNG named like "*Colormap*.png" under project root
    """
    value = config.get("colormap_png")
    if isinstance(value, str) and value:
        p = Path(value)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.exists():
            return p
    direct = (root / "Colormap.png").resolve()
    if direct.exists():
        return direct
    candidates = list(root.rglob("*Colormap*.png"))
    return candidates[0] if candidates else None


def pick_equidistant_colors(palette: Sequence[str], n: int) -> List[str]:
    if n <= 0:
        return []
    if len(palette) == 0:
        return []
    if len(palette) == 1:
        return [palette[0]] * n
    positions = np.linspace(0, len(palette) - 1, n)
    indices = np.clip(np.round(positions).astype(int), 0, len(palette) - 1)
    return [palette[i] for i in indices]


def get_display_label(config: Dict[str, Any], key: str, *, default: Optional[str] = None) -> str:
    labels = config.get("display_property_labels", {}) or {}
    return str(labels.get(key, default or key.replace("_", " ").title()))


def get_axis_label(config: Dict[str, Any], feature_name: str) -> str:
    labels = config.get("display_labels", {}) or {}
    return str(labels.get(feature_name, feature_name))


def get_category_label(config: Dict[str, Any], category_name: str, value: int) -> str:
    mapping = (config.get("display_categories", {}) or {}).get(category_name, {})
    return str(mapping.get(str(value), str(value)))


def benchmark_property_inference(
    kernel: Any,
    property_key: str,
    batch_sizes: Iterable[int],
    repeats: int = 50,
    warmup: int = 5,
) -> pd.DataFrame:
    """Benchmark per-sample latency (microseconds) across batch sizes.

    Args:
        kernel: Initialized `CGKernel`.
        property_key: e.g., "rms" or "growth_rate".
        batch_sizes: Iterable of batch sizes.
        repeats: Number of timed iterations per batch size.
        warmup: Number of warmup iterations (not timed).

    Returns:
        DataFrame with columns: batch_size (str), per_pred_us (float).
    """
    # Build a baseline sample within configured ranges
    cfg = kernel.config
    num_cols: List[str] = cfg.get("numeric_columns", []) + cfg.get("integer_columns", [])
    cat_col: str = cfg.get("categorical_column", "Nnucl")
    base: Dict[str, Any] = {}
    for fs in (cfg.get("features", []) or []):
        name = fs.get("name")
        if name in num_cols:
            vmin = float(fs.get("min", 0.0))
            vmax = float(fs.get("max", 1.0))
            base[name] = (vmin + vmax) / 2.0
        elif name == cat_col:
            base[name] = int(cfg.get("defaults", {}).get(cat_col, 0))

    rows: List[Dict[str, Any]] = []
    batch_sizes = list(batch_sizes)
    # Warm-up pass on smallest batch
    if batch_sizes:
        bs0 = max(1, int(batch_sizes[0]))
        samples = [base] * bs0
        _ = kernel.predict_property(property_key, samples)
    for bs in batch_sizes:
        bs = int(bs)
        if bs <= 0:
            continue
        samples = [base] * bs
        # warmup loops
        for _ in range(max(0, warmup)):
            _ = kernel.predict_property(property_key, samples)
        # timed loops
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            _ = kernel.predict_property(property_key, samples)
            dt = time.perf_counter() - t0
            rows.append({"batch_size": str(bs), "per_pred_us": (dt / float(bs)) * 1_000_000.0})
    return pd.DataFrame(rows)


def violin_latency_plot(df: pd.DataFrame, title: str, discrete_colors: Optional[List[str]] = None) -> go.Figure:
    categories = sorted(df["batch_size"].unique().tolist(), key=lambda x: int(x))
    fig = px.violin(
        df,
        x="batch_size",
        y="per_pred_us",
        box=True,
        points=False,
        color="batch_size",
        category_orders={"batch_size": categories},
        color_discrete_sequence=discrete_colors,
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Batch size",
        yaxis_title="Latency per prediction (µs)",
        width=1200,
        height=700,
        legend_title_text="Batch size",
    )
    fig.update_traces(meanline_visible=True, scalemode="width", width=0.9)
    fig.update_yaxes(type="log")
    return fig


def ensure_positive(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x2 = x.copy()
    x2[x2 < eps] = eps
    return x2


def build_meshgrid(df: pd.DataFrame, x_col: str, y_col: str, n: int, x_log: bool, y_log: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create X,Y vectors and 2D meshgrid respecting selected scales and percentile ranges."""
    x_min, x_max = percentile_range(df[x_col])
    y_min, y_max = percentile_range(df[y_col])
    if x_log:
        x_min = max(x_min, 1e-12)
        Xv = np.logspace(np.log10(x_min), np.log10(x_max), n)
    else:
        Xv = np.linspace(x_min, x_max, n)
    if y_log:
        y_min = max(y_min, 1e-12)
        Yv = np.logspace(np.log10(y_min), np.log10(y_max), n)
    else:
        Yv = np.linspace(y_min, y_max, n)
    Xg, Yg = np.meshgrid(Xv, Yv)
    return Xv, Yv, Xg, Yg


def make_discrete_step_colorscale(colors: List[str]) -> List[Tuple[float, str]]:
    k = len(colors)
    if k <= 1:
        c = colors[0] if colors else "#000000"
        return [(0.0, c), (1.0, c)]
    scale: List[Tuple[float, str]] = []
    for i, c in enumerate(colors):
        v0 = i / k
        v1 = (i + 1) / k
        scale.append((v0, c))
        scale.append((v1, c))
    return scale



def format_array_or_str(obj: Any, **kwargs) -> str:
    """Return a robust string for arrays while handling non-array inputs gracefully.

    This helper mirrors ``numpy.array2string`` formatting for ``numpy.ndarray``
    inputs and falls back to ``str(obj)`` for any other type. It is useful when
    upstream code may sometimes return a string instead of an array.

    Args:
        obj: Object to render; typically a ``numpy.ndarray`` or a string.
        **kwargs: Keyword arguments forwarded to ``numpy.array2string`` when
            ``obj`` is an ``np.ndarray`` (e.g., ``precision``, ``separator``).

    Returns:
        str: A human-readable representation of ``obj`` suitable for display.
    """
    return np.array2string(obj, **kwargs) if isinstance(obj, np.ndarray) else str(obj)


def get_presaved_morphology_parameters(class_id: int) -> Dict[str, Any]:
    """Get predefined parameters for a specific morphology class.
    
    This function returns the growth conditions that typically produce
    the specified morphology class. You can modify these values to match
    your specific dataset and requirements.
    
    Args:
        class_id: The morphology class ID (0-19).
        
    Returns:
        Dictionary with parameter names as keys and values as values.
    """
    # Default parameters - you can modify these values for each class
    default_params = {
        "Cs": 0.03,
        "Pes": 0.2,
        "Pbias": 0.03,
        "T": 0.50,
        "Pd": 0.0001,
        "M": 0,
        "Nnucl": 2,
    }
    
    # Class-specific parameter sets
    # You can customize these for each morphology class
    class_parameters = {
        0: {"Cs": 0.025233, "Pes": 0.296325, "Pbias": -0.067, "T": 0.2, "Pd": 0.00160788, "M": 14, "Nnucl": 1},
        1: {"Cs": 0.00117, "Pes": 0.971278, "Pbias": -0.037, "T": 0.82, "Pd": 0.00310291, "M": 14, "Nnucl": 2},
        2: {"Cs": 0.076045, "Pes": 0.850064, "Pbias": -0.27, "T": 0.18, "Pd": 2.95e-06, "M": 3, "Nnucl": 1},
        3: {"Cs": 0.063071, "Pes": 0.249615, "Pbias": -0.127, "T": 0.15, "Pd": 8.445e-05, "M": 7, "Nnucl": 2},
        4: {"Cs": 0.058918, "Pes": 0.124715, "Pbias": 0.266, "T": 0.69, "Pd": 5.21e-06, "M": 10, "Nnucl": 2},
        5: {"Cs": 0.025064, "Pes": 0.825814, "Pbias": -0.194, "T": 0.11, "Pd": 1.67e-06, "M": 10, "Nnucl": 0},
        6: {"Cs": 0.144654, "Pes": 0.002712, "Pbias": 0.172, "T": 0.72, "Pd": 0.00027181, "M": 0, "Nnucl": 2},
        7: {"Cs": 0.178498, "Pes": 0.091883, "Pbias": -0.278, "T": 0.3, "Pd": 4.23e-06, "M": 0, "Nnucl": 2},
        8: {"Cs": 0.020518, "Pes": 0.991437, "Pbias": 0.202, "T": 0.05, "Pd": 0.00011127, "M": 0, "Nnucl": 2},
        9: {"Cs": 0.130919, "Pes": 0.8238, "Pbias": 0.206, "T": 0.51, "Pd": 4.21e-06, "M": 0, "Nnucl": 2},
        10: {"Cs": 0.176731, "Pes": 0.814513, "Pbias": -0.249, "T": 0.22, "Pd": 0.00083011, "M": 14, "Nnucl": 2},
        11: {"Cs": 0.11621, "Pes": 0.471855, "Pbias": 0.3, "T": 0.96, "Pd": 0.00017355, "M": 10, "Nnucl": 1},
        12: {"Cs": 0.024106, "Pes": 0.547335, "Pbias": 0.226, "T": 0.74, "Pd": 0.00447959, "M": 3, "Nnucl": 2},
        13: {"Cs": 0.308574, "Pes": 0.159706, "Pbias": 0.167, "T": 0.44, "Pd": 3.46e-06, "M": 0, "Nnucl": 0},
        14: {"Cs": 0.082684, "Pes": 0.745467, "Pbias": -0.059, "T": 0.3, "Pd": 0.00037408, "M": 14, "Nnucl": 1},
        15: {"Cs": 0.137675, "Pes": 0.158998, "Pbias": 0.237, "T": 0.67, "Pd": 1.173e-05, "M": 14, "Nnucl": 2},
        16: {"Cs": 0.343139, "Pes": 0.323187, "Pbias": -0.067, "T": 0.74, "Pd": 1.36e-06, "M": 3, "Nnucl": 2},
        17: {"Cs": 0.478815, "Pes": 0.953175, "Pbias": -0.2, "T": 0.88, "Pd": 0.00011518, "M": 3, "Nnucl": 1},
        18: {"Cs": 0.474084, "Pes": 0.473608, "Pbias": -0.038, "T": 0.33, "Pd": 7.27e-06, "M": 0, "Nnucl": 2},
        19: {"Cs": 0.465835, "Pes": 0.302532, "Pbias": -0.163, "T": 0.5, "Pd": 0.00799203, "M": 10, "Nnucl": 2},
    }
    
    return class_parameters.get(class_id, default_params)


# Plotly configuration and helper functions for Google Colab and JupyterLab compatibility

def detect_environment() -> bool:
    """Detect if we're running in Google Colab.
    
    Returns:
        True if running in Google Colab, False otherwise.
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def configure_plotly():
    """Configure Plotly for the current environment (Google Colab or JupyterLab).
    
    This function sets up Plotly with the appropriate renderer and configuration
    for the detected environment.
    """
    import plotly.io as pio
    import plotly.offline as pyo
    
    in_colab = detect_environment()
    
    if in_colab:
        # Google Colab configuration
        pio.renderers.default = "colab"
        # Enable offline mode for Colab
        pyo.init_notebook_mode(connected=True)
        print("Plotly configured for Google Colab")
    else:
        # JupyterLab configuration (default)
        pio.renderers.default = "notebook"
        print("Plotly configured for JupyterLab")
    
    # Additional configuration for better compatibility
    pio.templates.default = "plotly_white"


def clear_output_safely(output_widget=None):
    """Clear output safely for both Google Colab and JupyterLab.
    
    Args:
        output_widget: Optional output widget to clear. If None, uses IPython.display.clear_output()
    """
    try:
        from IPython.display import clear_output
        if output_widget is not None:
            output_widget.clear_output(wait=True)
        else:
            clear_output(wait=True)
    except Exception:
        # Fallback: try to clear the output area
        try:
            import IPython
            if hasattr(IPython.get_ipython(), 'kernel'):
                IPython.get_ipython().kernel.do_shutdown(False)
        except Exception:
            pass


def show_plot(fig, **kwargs):
    """Display a Plotly figure with proper configuration for both Google Colab and JupyterLab.
    
    This function automatically detects the environment and uses the appropriate
    renderer for displaying Plotly figures.
    
    Args:
        fig: Plotly figure object
        **kwargs: Additional arguments to pass to fig.show()
    """
    in_colab = detect_environment()
    
    if in_colab:
        fig.show(renderer="colab", **kwargs)
    else:
        fig.show(**kwargs)


