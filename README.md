# Climate Residual Downscaling with Song U-Net

This repository provides a compact and reproducible workflow for residual-based climate downscaling using the Song U-Net architecture available in IPSL-AID.

The project is framed as a simplified inverse problem: given a spatially degraded 2-meter temperature field, the objective is to reconstruct the original field by learning the missing fine-scale residual.

The emphasis is on methodology, architecture, workflow design, and reproducibility rather than high predictive performance.

---

## 1. Conceptual formulation

Let:

- `y_hr` be the original high-resolution climate field.
- `y_smooth` be a spatially smoothed version of the same field.
- `r = y_hr - y_smooth` be the residual information removed by smoothing.

Instead of directly learning:

```text
y_hr = F(y_smooth)
```

the workflow learns the residual correction:

```text
r = G(y_smooth)
```

and reconstructs the field as:

```text
y_pred = y_smooth + r_pred
```

This formulation allows the neural network to focus on the missing spatial detail, while the large-scale atmospheric structure is already contained in the smoothed input field.

---

## 2. Dataset

The notebook uses ERA5 2-meter temperature data from WeatherBench2:

```text
gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr
```

Main characteristics:

- Variable: `2m_temperature`
- Physical meaning: near-surface air temperature at 2 meters
- Units: Kelvin
- Spatial grid: global latitude-longitude grid
- Grid size: `121 x 240`
- Approximate spatial resolution: `1.5° x 1.5°`
- Temporal resolution: 6 hours

The experiment uses a lightweight temporal sample to keep the workflow reproducible on a local CPU-based environment.

---

## 3. Method overview

The workflow follows these steps:

1. Load ERA5 2-meter temperature data.
2. Inspect the dataset structure, metadata, spatial resolution, temporal resolution, and physical units.
3. Compute descriptive statistics using both standard and area-weighted summaries.
4. Generate a degraded input field using Gaussian smoothing.
5. Define the residual field as the difference between the original and smoothed fields.
6. Train a Song U-Net model to predict the residual from the smoothed input.
7. Reconstruct the high-resolution field by adding the predicted residual to the smoothed field.
8. Evaluate the reconstruction against the degraded input baseline.
9. Save figures, tables, checkpoints, and reports under the `results/` directory.

---

## 4. Repository structure

```text
climate-residual-downscaling-unet/
├── README.md
├── notebooks/
│   └── 01_residual_downscaling_demo_v01_1.ipynb
├── results/
│   ├── figures/
│   ├── tables/
│   ├── reports/
│   └── checkpoints/
├── src/
└── tests/
```

The notebook is the main executable report. The `results/` directory stores generated figures, tables, exported reports, and model checkpoints.

The `src/` and `tests/` folders are included as placeholders for future modularization of the workflow into reusable Python functions and tests.

---

## 5. Installation

### 5.1 Get the project

If using Git:

```bash
git clone <your-repository-url>
cd climate-residual-downscaling-unet
```

If using the shared `.zip` file:

```bash
unzip climate-residual-downscaling-unet.zip
cd climate-residual-downscaling-unet
```

---

### 5.2 Create the Conda environment

```bash
conda create -n ipsl-aid python=3.11
conda activate ipsl-aid
```

Install the core scientific packages:

```bash
pip install numpy==1.26.4 pandas xarray zarr gcsfs matplotlib scipy scikit-learn
pip install jupyterlab notebook ipykernel nbconvert rich
```

Install PyTorch.

For the local CPU-based Mac environment used in this demo:

```bash
pip install torch==2.2.2 torchvision==0.17.2
```

For Linux, GPU, or HPC environments, install PyTorch using the official configuration appropriate for the target hardware.

---

### 5.3 Install IPSL-AID

Clone IPSL-AID separately:

```bash
cd ~
git clone https://github.com/kardaneh/IPSL-AID.git
cd IPSL-AID
```

Install it in editable mode without forcing all dependencies:

```bash
pip install -e . --no-deps
```

Return to this project:

```bash
cd ~/climate-residual-downscaling-unet
```

Check the installation:

```bash
python -c "import IPSL_AID; print('IPSL_AID ok')"
python -c "from IPSL_AID.networks import SongUNet; print('SongUNet ok')"
```

---

### 5.4 Register the Jupyter kernel

```bash
python -m ipykernel install --user --name ipsl-aid --display-name "Python (ipsl-aid)"
```

---

## 6. Running the notebook

Start JupyterLab from the project root:

```bash
conda activate ipsl-aid
cd ~/climate-residual-downscaling-unet
jupyter lab
```

Open:

```text
notebooks/01_residual_downscaling_demo_v01_1.ipynb
```

Select the kernel:

```text
Python (ipsl-aid)
```

Then run the notebook from top to bottom.

The notebook automatically saves generated outputs into:

```text
results/figures/
results/tables/
results/checkpoints/
results/reports/
```

---

## 7. Outputs

The workflow produces:

```text
results/figures/
├── initial ERA5 temperature map
├── temperature distribution
├── Gaussian smoothing and residual visualization
├── training history
├── reconstruction comparison
├── residual comparison
└── error comparison
```

```text
results/tables/
├── dataset statistics
├── residual statistics
├── training history
├── test metrics
└── relative improvement
```

```text
results/checkpoints/
└── Song U-Net model checkpoint
```

```text
results/reports/
└── exported HTML/PDF report
```

---

## 8. Exporting the report

The notebook includes an optional export cell at the end.

Before running the export cell, save the notebook manually:

```text
File -> Save Notebook
```

The HTML report is saved under:

```text
results/reports/
```

PDF export through `webpdf` may require Playwright and Chromium:

```bash
pip install "nbconvert[webpdf]"
python -m playwright install chromium
```

After that, rerun the final export cell or export manually:

```bash
jupyter nbconvert \
  --to webpdf \
  notebooks/01_residual_downscaling_demo_v01_1.ipynb \
  --output-dir results/reports
```

If PDF export fails, open the HTML report in a browser and use:

```text
Print -> Save as PDF
```

---

## 9. Notes on training

The Song U-Net model is trained for only 3 epochs in this demonstration.

This is intentional. The goal is to demonstrate a reproducible residual-learning workflow, not to maximize predictive performance. The experiment was designed to run on a local CPU-based environment. Longer training, larger temporal sampling, hyperparameter tuning, GPU/HPC execution, and extended baselines would be required for a performance-oriented benchmark.

The decreasing training and validation losses indicate that the implementation is operational and that the model is learning a residual correction.

---

## 10. Relation to IPSL-AID

This repository uses the Song U-Net architecture from IPSL-AID, but implements a simplified residual-learning workflow.

The IPSL-AID framework uses a more complete downscaling strategy based on coarse-up fields, residual generation, and diffusion-based modelling. In this repository, Gaussian smoothing is used as a controlled image-processing degradation operator to create a lightweight and interpretable proof-of-workflow.

Conceptually:

```text
This notebook:
y_smooth -> Song U-Net -> residual prediction -> reconstruction

IPSL-AID diffusion framework:
y_CU + noise -> diffusion model -> residual generation -> reconstruction
```

Both approaches are based on the idea that the degraded field contains the large-scale climate structure, while the residual contains the missing fine-scale spatial variability.

---

## 11. Limitations and extensions

This repository provides a compact proof-of-workflow.

Main limitations:

- The experiment uses a lightweight temporal sample rather than the full ERA5 time series.
- Training is intentionally short and CPU-compatible.
- The current model is deterministic and predicts a single residual correction.
- The degradation operator is Gaussian smoothing, while the full IPSL-AID framework uses a coarse-up strategy based on resolution degradation and interpolation.
- The loss function is a standard pixel-wise residual loss.

Natural extensions include:

- training on a larger temporal dataset;
- running on GPU or HPC infrastructure;
- adding multiple input variables such as humidity, wind, pressure, radiation, or precipitation;
- adding static covariates such as orography, land-sea mask, latitude, and longitude;
- comparing against interpolation methods, shallow CNNs, non-residual U-Nets, and diffusion-based baselines;
- evaluating spatial spectra, residual variance, gradients, and regional/seasonal performance;
- extending the workflow toward probabilistic or diffusion-based residual generation.

---

## 12. Purpose

This repository is intended as a compact, inspectable, and extensible research workflow for climate residual downscaling.

It demonstrates how a Song U-Net architecture can be integrated into a reproducible pipeline for residual reconstruction of gridded climate fields.