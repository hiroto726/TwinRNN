# TwinRNN 

This repository contains code associated with the paper:

**Independence and Coherence in Temporal Sequence Computation across the Fronto-Parietal Network.**

The main entry point is a Jupyter notebook that loads a pretrained Twin RNN model (stored in `RNN_models/`) and reproduces activity under perturbation.
When executed successfully, the generated output should qualitatively reproduce the activity pattern shown in **Figure 5K** of the paper.



---

## To Start

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/TwinRNN.git
cd TwinRNN
```
### 2. Set up the environment
Configure the environment using either Docker (recommended), pip, or conda. For details, see below.

### 3. Launch Jupyter

Always launch Jupyter from the repository root directory so relative paths resolve correctly.

```bash
jupyter notebook
```

Open `notebooks/visualize_activity.ipynb` and run all cells from top to bottom.

---

## Environment Setup

You can reproduce the environment using one of the following methods.

---

### Option A: Docker (Recommended for exact reproducibility)

Build the Docker image:

```bash
docker build -t twinrnn:latest .
```

Run the container and launch Jupyter:

```bash
docker run --rm -it -p 8888:8888 -v "%cd%":/workspace twinrnn:latest \
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Open the URL displayed in the terminal (it includes an access token).

Notes:
- Based on TensorFlow 2.1.0 GPU (CUDA 10.1 era).
- GPU support requires NVIDIA GPU + NVIDIA Container Toolkit.

---

### Option B: pip (Local Python Environment)

Use Python 3.6 or 3.7. TensorFlow 2.1.0 does not provide wheels for newer Python versions such as Python 3.11.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

---

### Option C: conda

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it and launch Jupyter:

```bash
conda activate twinrnn
jupyter notebook
```

On Windows, if directly running the environment's `python.exe` gives NumPy/MKL DLL errors, run commands from an activated conda prompt or use `conda run -n twinrnn ...`.

---

## Expected Output

The notebook should:

1. Load the pretrained Twin RNN model from `RNN_models/`
2. Run the model under perturbation conditions
3. Display the resulting network activity

The notebook imports helper code from `functions/`.

---

## Optional Validation

To execute the notebook from the command line and confirm that all cells run:

```bash
jupyter nbconvert --to notebook --execute notebooks/visualize_activity.ipynb --output visualize_activity.executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=3600
```

Generated validation notebooks and figures matching `notebooks/*.executed.ipynb` and `notebooks/*.figure.png` are ignored by Git.

---

## Important Notes

- Tested with:
  - Python 3.6.13 (`myRNN3`) and Python 3.7-compatible environment files
  - TensorFlow 2.1.0
  - Keras 2.3.1
  - NumPy 1.18.5
  - SciPy 1.4.1
  - Protobuf 3.20.3
- If running outside Docker, use Python 3.6 or 3.7.
- GPU is optional for the demo notebook. Without CUDA 10.1 and cuDNN 7, TensorFlow 2.1.0 should fall back to CPU.
- Always run notebooks from the repository root directory.

---
