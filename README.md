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

### Step 3: Install the 'meltingpot' submodule
```bash
cd meltingpot
pip install --editable .[dev]
cd ..
```

### Step 4: Install Core MARL Tools
```bash
pip install torchrl
```

### Step 5: Install the 'BenchMARL' submodule
```bash
pip install -e BenchMARL
```

### Step 6: Install Evaluation Tools
```bash
pip install matplotlib
pip install id-marl-eval
```

## Usage
To train a single experiment on Coins environment without asymmetry run the script below.

```bash
python run_experiment.py -m task=meltingpot/asymmetric_coins__default seed=0
```

To change experiment settings, modify the necessary YAML files under ssd_config folder.


## 🌎 Environment Configurations and Versions

Our experiments utilize two base environments, **Coins** and **Harvest**, each adapted to explore different forms of asymmetry. The specific reward function modifications (Incentive Structures) and agent types define each unique environment version.

### 1. Agent Types

The agent types used in the environments introduce specific asymmetries in rewards and/or abilities:

| **Agent Type** | **Reward per Item** | **Abilities / Effects** | **Environment(s)** |
| :--- | :--- | :--- | :--- |
| **Standard** | $+1.0$ | Movement; zap in **Harvest** only | Harvest, Coins |
| **Low-reward** | $+0.5$ | Movement; in **Coins**, mismatch coin penalizes other by $-3.0$ | Harvest, Coins |
| **High-reward** | $+1.5$ | Movement; in **Coins**, mismatch coin penalizes other by $-1.0$ | Harvest, Coins |
| **Wide-zap** | $+1.0$ | Larger zapping radius | Harvest |
| **Spawn-biased** | $+1.0$ | Mismatch coin collection makes **only this agent's** coins spawn for 25 steps | Coins |

---

### 2. Incentive Structures (Reward Function Suffixes)

The suffix of the configuration file indicates the type of social incentive structure applied to the agents' reward functions:

| Suffix | Full Name | Description |
| :--- | :--- | :--- |
| **IA** | Inequity Aversion | Agents are penalized for large differences in observed utility. |
| **SVO** | Social Value Orientation | Agents incorporate the utility of the other agent into their own reward function. |
| **flia** | Fair & Local IA | A local and fairness-aware version of Inequity Aversion. |
| **flsvo** | Fair & Local SVO | A local and fairness-aware version of Social Value Orientation. |
| **\_both\_coop** | Both Agents Cooperative | Both agents use the standard cooperative baseline reward structure. |

---

### 3. Environment Versions (Example Configurations)

The configurations listed below demonstrate the combination of **Asymmetry Type** and **Incentive Structure**. Files containing `default` refer to the **symmetric** version of the environment.

| Asymmetry Type | Incentive Structure | Config File |
| :--- | :--- | :--- |
| **Symmetric** (Default) | Standard | `asymmetric_coins_default.yaml` |
| **Symmetric** (Default) | Fair&Local IA (flia) | `asymmetric_coins_default_flia.yaml` |
| **1 High-Reward, 1 Low-Reward** | Both Coop | `asymmetric_coins_1high_1low_reward_both_coop.yaml` |
| **1 High-Reward, 1 Low-Reward** | Fair&Local IA (flia) | `asymmetric_coins_1high_1low_reward_flia.yaml` |
| **1 High-Reward, 1 Low-Reward** | Inequity Aversion (IA) | `asymmetric_coins_1high_1low_reward_ia.yaml` |
| **1 Standard, 1 Spawn-Biased** | Both Coop | `asymmetric_coins_1standard_1spawn_biased_both_coop.yaml` |
| **1 Standard, 1 Spawn-Biased** | Fair&Local IA (flia) | `asymmetric_coins_1standard_1spawn_biased_flia.yaml` |
| **1 Standard, 1 Spawn-Biased** | Social Value Orientation (SVO) | `asymmetric_coins_1standard_1spawn_biased_svo.yaml` |
