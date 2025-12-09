# ✨ Fairness in SSD: Consolidated MARL Environment Setup

This repository contains the primary code for "Fairness in SSD" and uses Git Submodules to manage key external dependencies: `meltingpot` and `BenchMARL`.

## ⚙️ Prerequisites

This installation guide relies on **Conda** for managing the Python environment. Please ensure Conda is installed on your system.

---

## 💻 Installation Guide

### Step 1: Clone the Main Repository and Submodules

To clone this repository and automatically initialize and fetch the contents of the linked submodules, use the following commands.

```bash
# Clone the main repository, including all submodules recursively
git clone --recurse-submodules [https://github.com/gliese876b/fairness_in_ssd.git](https://github.com/gliese876b/fairness_in_ssd.git)
cd fairness_in_ssd
```

### Step 2: Create and Activate the Conda Environment

We use Conda to create a clean, isolated environment with Python 3.11.

```bash
conda create -n bench_env python=3.11
conda activate bench_env
```

# 1. Install the 'meltingpot' submodule
```bash
cd meltingpot
pip install --editable .[dev]
cd ..
```

# 2. Install Core MARL Tools
```bash
pip install torchrl
```

# 3. Install the 'BenchMARL' submodule
# The submodule is already cloned into the BenchMARL/ directory.
```bash
pip install -e BenchMARL
```

# 4. Install Evaluation Tools
```bash
pip install matplotlib
pip install id-marl-eval
```
