# Diffusion Policy Experiments

Michael's personal diffusion policy experiments/implementation/research in the Robot Locomotion Group, based off of [original diffusion policy paper](https://diffusion-policy.cs.columbia.edu/) and work done by Abhinav Agarwal and Adam Wei.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites
- Python 3.10
- Poetry (for dependency management)

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

5. **Install dependencies**:
```bash
poetry install
```

6. **Activate the Poetry environment**:
```bash
source $(poetry env info --path)/bin/activate
```



## Running

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
