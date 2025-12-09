# ✨ Fairness in SSD: Consolidated MARL Environment Setup

This repository contains the primary code for "Fairness over Equality: Correcting Social Incentives in Asymmetric Sequential Social Dilemmas" and uses Git Submodules to manage key external dependencies: `meltingpot` and `BenchMARL`.

## ⚙️ Prerequisites

This installation guide relies on **Conda** for managing the Python environment. Please ensure Conda is installed on your system.

---

## 💻 Installation Guide

### Step 1: Clone the Main Repository and Submodules

To clone this repository and automatically initialize and fetch the contents of the linked submodules, use the following commands.

```bash
# Clone the main repository, including all submodules recursively
git clone --recurse-submodules https://github.com/gliese876b/fairness_in_ssd.git
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

To train a single experiment of Fair&LocalSVO on Harvest with asymmetry in apple rewards run the script below.

```bash
python run_experiment.py -m task=meltingpot/asymmetric_commons_harvest_5high_5low_reward_flsvo seed=0
```

To change experiment settings, modify the necessary YAML files under `ssd_config` folder.


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
| **IA** | Inequity Aversion | see [the paper](https://proceedings.neurips.cc/paper/2018/file/7fea637fd6d02b8f0adf6f7dc36aed93-Paper.pdf) |
| **SVO** | Social Value Orientation | see [the paper](https://www.ifaamas.org/Proceedings/aamas2020/pdfs/p869.pdf) |
| **flia** | Fair & Local IA | Our proposed Fair&Local version of IA |
| **flsvo** | Fair & Local SVO | Our proposed Fair&Local version of SVO |

---

### 3. Environment Versions (Example Configurations)

The configurations listed below demonstrate the combination of **Asymmetry Type** and **Incentive Structure**. Files containing `default` refer to the **symmetric** version of the environment.

### 3. Environment Versions (from Config Files)

The configurations below detail the specific combinations of **Asymmetry Type** and **Incentive Structure** used across the **Coins** and **Harvest** environments. Files containing `default` refer to the **symmetric** version of the environment.

| Environment | Asymmetry Type | Incentive Structure | Config File |
| :--- | :--- | :--- | :--- |
| **Coins** | Symmetric (Default) | Standard | `asymmetric_coins_default.yaml` |
| **Coins** | Symmetric (Default) | Fair&Local IA (flia) | `asymmetric_coins_default_flia.yaml` |
| **Coins** | Symmetric (Default) | Fair&Local SVO (flsvo) | `asymmetric_coins_default_flsvo.yaml` |
| **Coins** | Symmetric (Default) | Inequity Aversion (IA) | `asymmetric_coins_default_ia.yaml` |
| **Coins** | Symmetric (Default) | Social Value Orientation (SVO) | `asymmetric_coins_default_svo.yaml` |
| **Coins** | **1 High-Reward, 1 Low-Reward** | Standard | `asymmetric_coins_1high_1low_reward.yaml` |
| **Coins** | **1 High-Reward, 1 Low-Reward** | Fair&Local IA (flia) | `asymmetric_coins_1high_1low_reward_flia.yaml` |
| **Coins** | **1 High-Reward, 1 Low-Reward** | Fair&Local SVO (flsvo) | `asymmetric_coins_1high_1low_reward_flsvo.yaml` |
| **Coins** | **1 High-Reward, 1 Low-Reward** | Inequity Aversion (IA) | `asymmetric_coins_1high_1low_reward_ia.yaml` |
| **Coins** | **1 High-Reward, 1 Low-Reward** | Social Value Orientation (SVO) | `asymmetric_coins_1high_1low_reward_svo.yaml` |
| **Coins** | **1 Standard, 1 Spawn-Biased** | Standard | `asymmetric_coins_1standard_1spawn_biased.yaml` |
| **Coins** | **1 Standard, 1 Spawn-Biased** | Fair&Local IA (flia) | `asymmetric_coins_1standard_1spawn_biased_flia.yaml` |
| **Coins** | **1 Standard, 1 Spawn-Biased** | Fair&Local SVO (flsvo) | `asymmetric_coins_1standard_1spawn_biased_flsvo.yaml` |
| **Coins** | **1 Standard, 1 Spawn-Biased** | Inequity Aversion (IA) | `asymmetric_coins_1standard_1spawn_biased_ia.yaml` |
| **Coins** | **1 Standard, 1 Spawn-Biased** | Social Value Orientation (SVO) | `asymmetric_coins_1standard_1spawn_biased_svo.yaml` |
| **Harvest** | Symmetric (Default) | Standard | `asymmetric_commons_harvest_default.yaml` |
| **Harvest** | Symmetric (Default) | Inequity Aversion (IA) | `asymmetric_commons_harvest_default_ia.yaml` |
| **Harvest** | Symmetric (Default) | Social Value Orientation (SVO) | `asymmetric_commons_harvest_default_svo.yaml` |
| **Harvest** | **5 High-Reward, 5 Low-Reward** | Standard | `asymmetric_commons_harvest_5high_5low_reward.yaml` |
| **Harvest** | **5 High-Reward, 5 Low-Reward** | Fair&Local IA (flia) | `asymmetric_commons_harvest_5high_5low_reward_flia.yaml` |
| **Harvest** | **5 High-Reward, 5 Low-Reward** | Fair&Local SVO (flsvo) | `asymmetric_commons_harvest_5high_5low_reward_flsvo.yaml` |
| **Harvest** | **5 High-Reward, 5 Low-Reward** | Inequity Aversion (IA) | `asymmetric_commons_harvest_5high_5low_reward_ia.yaml` |
| **Harvest** | **5 High-Reward, 5 Low-Reward** | Social Value Orientation (SVO) | `asymmetric_commons_harvest_5high_5low_reward_svo.yaml` |
| **Harvest** | **5 Standard, 5 Wide-Zapper** | Standard | `asymmetric_commons_harvest_5standard_5wide_zapper.yaml` |
| **Harvest** | **5 Standard, 5 Wide-Zapper** | Fair&Local IA (flia) | `asymmetric_commons_harvest_5standard_5wide_zapper_flia.yaml` |
| **Harvest** | **5 Standard, 5 Wide-Zapper** | Fair&Local SVO (flsvo) | `asymmetric_commons_harvest_5standard_5wide_zapper_flsvo.yaml` |
| **Harvest** | **5 Standard, 5 Wide-Zapper** | Inequity Aversion (IA) | `asymmetric_commons_harvest_5standard_5wide_zapper_ia.yaml` |
| **Harvest** | **5 Standard, 5 Wide-Zapper** | Social Value Orientation (SVO) | `asymmetric_commons_harvest_5standard_5wide_zapper_svo.yaml` |
