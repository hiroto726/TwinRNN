# TwinRNN 

This repository contains code associated with the paper:

**“Independence and Coherence in Temporal Sequence Computation across the Fronto-Parietal Network.”**

The main entry point is a Jupyter notebook that loads a pretrained Twin RNN model (stored in `RNN_models/`) and reproduces activity under perturbation.
When executed successfully, the generated output should qualitatively reproduce the activity pattern shown in **Figure 5K** of the paper.



---

## To Start

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/TwinRNN.git
cd TwinRNN
```

### 2. Launch Jupyter

Always launch Jupyter from the repository root directory so relative paths resolve correctly.

```bash
jupyter notebook
```

Open the notebook in `notebooks/` and run all cells from top to bottom.

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

Activate it and Launch Jupyter

---

## Expected Output

The notebook should:

1. Load the pretrained Twin RNN model from `RNN_models/`
2. Run the model under perturbations
3. Display model activity

---

## Important Notes

- Tested with:
  - Python 3.7
  - TensorFlow 2.1.0
  - Keras 2.3.1
- If running outside Docker, ensure Python 3.7 is used.
- Always run notebooks from the repository root directory.

---
