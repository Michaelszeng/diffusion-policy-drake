# Diffusion Policy Experiments

Michael's personal diffusion policy experiments/implementation/research in the Robot Locomotion Group, based off of [original diffusion policy paper](https://diffusion-policy.cs.columbia.edu/) and work done by Abhinav Agarwal and Adam Wei.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites
- Python 3.9 or 3.10
- Poetry (for dependency management)

### Local Installation:

1. **Install Poetry** (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Add Poetry to your PATH**:
```bash
export PATH="/home/$USER/.local/bin:$PATH"
```

3. **Modify `pyproject.toml`**:
For the project to run the diffusion policy, it must have access to the diffusion policy repo as well as a trained model checkpoint. Clone [my diffusion policy repo](https://github.com/Michaelszeng/diffusion-policy-experiments), follow the instructions to run training for the planar pushing example, then modify this line in `pyproject.toml` to point to wherever you cloned your repo:

```
diffusion-policy = {path = "/home/michzeng/diffusion-policy", develop = true}
```

4. **If using a custom installation of drake**:

Add the following line in the `$(poetry env info --path)/bin/activate`:

```bash
export PYTHONPATH={PATH_TO_DRAKE_BUILD}/drake_build/lib/python3.10/site-packages:${PYTHONPATH}
```

Otherwise, add this to your `project.toml` file to use the latest pip wheel of drake:

```
drake = "1.45.0"
```

5. Install MOSEK
Download [MOSEK](mosektoolslinux64x86.tar.bz2) and unzip it in your home directory. Obtain an academic license. Place `mosek.lic` file in your home directory. Add this to your `~/.bashrc`: 

```
export MOSEKLM_LICENSE_FILE=~/mosek.lic
export LD_LIBRARY_PATH=$HOME/mosek/11.0/tools/platform/linux64x86/bin:$LD_LIBRARY_PATH
```


6. **Install dependencies**:
```bash
poetry install
```

7. **Activate the Poetry environment**:
```bash
source $(poetry env info --path)/bin/activate
```

### Supercloud Installation:
```bash
# module load anaconda/2023b
module load anaconda/Python-ML-2024b  # This module contains a lot of the dependencies we need
# Now, we install the remaining dependencies we need
pip install drake --no-deps  # TODO: instructions for local drake build?
pip install manipulation==2025.1.3 --no-deps
pip install huggingface-hub==0.25.2 --no-deps
pip install diffusers==0.11.1 --no-deps
pip install numba==0.60.0
pip install opencv-python==4.9.0.80 --no-deps
pip install robomimic==0.3.0 --no-deps
pip install hydra-core
pip install wandb
pip install einops
pip install zarr
pip install lxml
pip install lxml-html-clean
pip install pydot
pip install mpld3
pip install pyvirtualdisplay

pip install -e /home/gridsan/mzeng/diffusion-policy-experiments --no-deps
```

8. Setting up Supercloud Running Scripts

`scp` MOSEK license file to SuperCloud.

Modify the path to your license file in `submit_run_sim_sim_eval.sh` and `submit_launch_eval.sh`:

```bash
export MOSEKLM_LICENSE_FILE=/home/gridsan/mzeng/mosek.lic
```



## Running

### Running Locally

Parallel Experiments:
```bash
python scripts/launch_eval.py \
    --csv-path config/main_launch_eval.txt \
    --max-concurrent-jobs 8 \
    --num-trials 50 50 100 \
    --drop-threshold 0.05
```

Single Experiment:
```bash
python scripts/run_sim_sim_eval.py --config-dir=config/sim_config/sim_sim --config-name=gamepad_teleop_carbon 'diffusion_policy_config.checkpoint="/home/michzeng/diffusion-policy/data/outputs/planar_pushing/2_obs/checkpoints/latest.ckpt"'
```

### Running on Supercloud:
```bash
# Interactively:
LLsub -i -s 20 -g volta:1
module load anaconda/2023b
wandb offline  # Supercloud compute nodes have no internet
python scripts/launch_eval.py \
    --csv-path config/main_launch_eval.txt \
    --max-concurrent-jobs 8 \
    --num-trials 50 50 100 \
    --drop-threshold 0.05

# or:
python -m scripts.run_sim_sim_eval --config-dir=config/sim_config/sim_sim --config-name=gamepad_teleop_carbon 'diffusion_policy_config.checkpoint="/home/gridsan/mzeng/diffusion-policy-experiments/data/outputs/planar_pushing/2_obs/checkpoints/latest.ckpt"'

# Non-interactively:
LLsub ./submit_eval.sh -s 20 -g volta:1
```

