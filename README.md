# Crystal Growth Kernel: Interactive Demo and Benchmarks

This repository demonstrates the AI-based approach developed by <b>A.V. Redkov</b> and <b>D.I. Trotsenko</b>, as detailed in the paper <b><i>"When Epitaxy Meets AI: Toward All-In-Silico Technology Design"</i></b>. The kernel is a state-of-the-art digital library consisting of AI models trained on a large dataset of high-throughput atomistic simulations, designed to predict the epitaxial growth process with unparalleled speed and accuracy.

Repository Contents:
- <b>Trained Crystal Growth Kernel</b>: This AI model predicts several critical properties related to epitaxial growth, including growth rate, surface morphology, roughness (RMS and average), kurtosis, peak-to-valley, skewness, and growth stability. It offers rapid predictions based on growth conditions, leveraging a comprehensive dataset of 299,000 simulated epitaxial growth experiments, totaling 3.2 TB of data.

- <b>Demo Jupyter Notebook</b>: This notebook provides an interactive demonstration of the kernel's capabilities. Users can explore how growth properties depend on conditions, generate morphologies for specific scenarios, and visualize the results interactively. Use sliders to adjust parameters, view 2D and 3D plots, and explore the multidimensional structure and stability zones across all epitaxial growth regimes—from step-flow to nucleation and various surface instabilities.

## Key Features:

- Instant Prediction: The kernel provides fast, accurate predictions of growth properties without the need for expensive atomistic simulations.
- Surface Morphology Generation: Instantly generates AFM-like simulated images based on growth conditions.
- Growth Type Classification: Classifies possible types of growth and maps them to growth condition parameters.
- Multidimensional Mapping: Visualizes stability zones, highlighting parameter regions where specific growth types occur.

This repository demonstrates a kernel trained on a dataset generated using a simplified atomistic model for the epitaxial growth of a cubic Kossel crystal from its own vapor. The model incorporates seven key parameters—such as adatom concentration, desorption probability, and step transparency—that govern growth dynamics, as detailed in the accompanying publication.

<i>*Note: This kernel, trained on a simplified growth model, serves as a proof of concept. The framework is adaptable to more complex atomistic models for real-world processes such as MBE and MOCVD. The training framework is available upon request.</i>

## Installation
1) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
```

2) Install runtime dependencies
```bash
pip install -r requirements.txt
```

- If you need GPU acceleration (the preferred way), install a CUDA-enabled `torch` build from the PyTorch site.

## Main demonstration
- The main demonstration is the `notebooks/cgkernel_demo.ipynb` notebook. It contains examples of all key functions: property regression, morphology and stability classification, trained-range validation behavior, and GAN image generation.

## Google Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avredkov/CG-Kernel/blob/main/notebooks/cgkernel_demo.ipynb)

## Project structure
- `cgkernel.py`: core implementation of `CGKernel`
- `cgkernel_config.json`: input schema, display labels, model files, and ranges
- `models/`: trained artifacts (`*.pt`, `*.preprocessor.pkl`); optional `generator.ts`
- `morphologies/`: reference PNGs used by `show_moprhology_classes()`
- `data/all_data299k.csv`: the file with the dataset, obtained using the atomistic model (see paper)
- `ml/`, `mlc/`: minimal preprocessor stubs used during inference
- `notebooks/cgkernel_demo.ipynb`: MAIN demonstration notebook with examples of all functions

## Input schema and ranges
The kernel reads the schema from `cgkernel_config.json`. Canonical feature names ('growth conditions') and ranges:

| Name   | DType        | Scale  | Min        | Max     | Notes |
|--------|--------------|--------|------------|---------|-------|
| Cs     | float        | log    | 0.001      | 0.5     | Surface concentration of adatoms. Must be > 0 (log10) 
| Pes    | float        | linear | 0.000001   | 1       | Terrace-to-terrace hopping probability
| Pbias  | float        | linear | -0.3       | 0.3     | Probability related to adatom drift in direction towards or opposite relative to step propagation
| T      | float        | linear | 0.0        | 1.0     | Step transparency (probability of adatom to incorporate into the surface in the kink position)
| Pd     | float        | log    | 0.000001   | 0.01    | Desorption probability.  Must be > 0 (log10)
| M      | int          | linear | 0          | 14      | Number of threading dislocations per area. Treated as integer
| Nnucl  | categorical  | linear | 0          | 2       | Parameter, describing allowed surface events (0 - kink incorporation only, 1 - 1D-nucleation on steps is also allowed, 2 - 2D nucleation on terraces is also allowed)

Display labels and category names are also provided via the config and used in printed summaries.

*<i>Note that the current kernel was trained using the set of parameters specified above. However, the training framework is designed to accommodate an arbitrary number and variety of input and output parameters, both continuous, integer and categorical and can be adjusted to suit different growth scenarios. This flexibility enables adaptation of the kernel to a wide range of atomistic growth models, including those relevant to real-world processes such as molecular beam epitaxy (MBE), metal-organic chemical vapor deposition (MOCVD), and others.</i>

### Trained-range validation
All prediction APIs validate that inputs lie within the configured ranges above. For features listed under `log_columns` (currently `Cs`, `Pd`), values must be strictly positive (log10 applied internally).

Out-of-range handling:
- Continuous predictions return the string: `"Parameter set is out of trained range"`.
- Classifier predictions return the tuple: `("Parameter set is out of trained range", None)`.

## Usage
```python
from cgkernel import CGKernel

# Keep `models/`, `morphologies/`, and `cgkernel_config.json` together in the folde <config_dir>
kernel = CGKernel(config_dir=".")
kernel.describe()

sample = {
    "Cs": 0.12,      # Surface concentration (was "Coverage")
    "Pes": 0.15,     # Schwoebel barrier (was "Es")
    "Pbias": 0.0,    # Drift of adatoms (was "Bias X")
    "T": 0.6,        # Transparency of steps (was "Pg")
    "Pd": 1e-3,      # Desorption probability (was "Desorption")
    "M": 10,         # Void number (was "N pits")
    "Nnucl": 1,      # Nucleation regime (was "Regime")
}

print(kernel.predict_property("rms", sample))
print(kernel.predict_property("ra", [sample, {**sample, "Cs": 0.2}]))
print(kernel.predict_morphology_class(sample))
print(kernel.predict_stability(sample))

kernel.show_moprhology_classes()
```

Notes:
- You can pass a single dict, a list of dicts, or a `pandas.DataFrame`.
- Aliases from older code and notebooks are supported and mapped to the new canonical names internally.

## GAN morphology synthesis
The `generate_morphology()` can synthesize surface images based on the same input features used by classifiers/regressors.

How to enable:
- Place a TorchScript file at `models/generator.ts`, or
- Set in `cgkernel_config.json` under `generator`:
  - `"pth"`: path to weights (TorchScript preferred)
  - Or `"import"` and optional `"module_path"` to import a Python `Generator` class and load a `state_dict`

Example:
```python
imgs = kernel.generate_morphology(sample, seed=42, scale_by_peak_to_valley=True)
# imgs is (H, W) float16 for a single sample
```

## CLI and logging
- There is no CLI at present; use the Python API as above.
- Adjust logging verbosity in your app:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Troubleshooting
- If the plots are empty in Google Colab runtime: ensure the Plotly version 5.10 is installed (launch the first cell in the notebook), as version 5.24 (currently default) is not supported.
- "Parameter set is out of trained range": ensure all numeric values are within the min/max listed above; for `Cs` and `Pd` values must be strictly > 0.
- Missing required columns: pass all features (`Cs`, `Pes`, `Pbias`, `T`, `Pd`, `M`, `Nnucl`). Aliases are accepted.
- Model files not found: confirm the `models/` directory contains the `*.pt` and matching `*.preprocessor.pkl` files listed in `cgkernel_config.json`.
- GAN not loaded: provide `models/generator.ts` or a valid `generator.import` reference.

## License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](./LICENSE.txt) file for details.

## How to Contribute

Feel free to fork this project and submit issues or pull requests. Contributions are always welcome!
