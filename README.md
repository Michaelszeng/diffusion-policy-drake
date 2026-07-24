# Planar Pushing with Diffusion Policy in Drake

Experimentation platform for planar pushing tasks in the [Drake](https://drake.mit.edu/) simulator. Based off work done by Abhinav Agarwal and Adam Wei in [[1](https://arxiv.org/abs/2503.22634)].

This repo contains the planar pushing environments. They can be used for things like teleop data collection, Graph-of-Convex-Sets (GCS) expert automated data collection, and diffusion policy evaluation. For diffusion policy evaluation, this repo integrates with the training repo [diffusion-policy-experiments](https://github.com/Michaelszeng/diffusion-policy-experiments).

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
To run evaluation of policies trained using, [diffusion-policy-experiments](https://github.com/Michaelszeng/diffusion-policy-experiments), clone [diffusion-policy-experiments](https://github.com/Michaelszeng/diffusion-policy-experiments), follow the instructions to run training for the planar pushing example, then modify this line in `pyproject.toml` to point to wherever you cloned the repo:

```
diffusion-policy = {path = "/home/michzeng/diffusion-policy-experiments", develop = true}
```

To run the GCS planner for automated data generation, the project must have access to my fork of the `planning-through-contact` repository. Clone [my `planning-through-contact` repo](https://github.com/Michaelszeng/planning-through-contact), then modify this line in `pyproject.toml` to point to wherever you cloned the repo:

```
gcs-planner = {path = "/home/michzeng/planning-through-contact", develop = true}
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

For installation in other environments (i.e. SLURM clusters or MIT/Lincoln Labs' SuperCloud), see `README_SLURM.md` or `README_MIT_SUPERCLOUD.md`.


## Running Teleop Data Collection

Use `scripts/run_gamepad_teleop.py` to collect demonstrations by teleoperating the pusher with a Logitech G F310 gamepad connected via Meshcat's browser Gamepad API:

```bash
python scripts/run_gamepad_teleop.py --config-name sim_sim/gamepad_teleop_carbon
```

Controls: `A` starts/saves a recording, `B` discards the current recording and resets, `X` quits, `RT`/`LT` halve/triple the movement speed. Set `data_collection_config.convert_to_zarr=true` (in the config or as a CLI override) to automatically convert the collected trajectories to a zarr dataset once the session ends.



## Running Diffusion Policy Evaluation

### Setting up the Eval Script

To run parallel evaluations, we launch `scripts/launch_eval.py` with a provided CSV config file with the following format:

```csv
checkpoint_path,run_dir,config_name,overrides
```

- **checkpoint_path**: Path to checkpoint file (.ckpt) or directory containing checkpoints
- **run_dir**: Output directory for evaluation results
- **config_name**: (Optional) Config file to use (defaults to `gamepad_teleop.yaml`)
- **overrides**: (Optional) Hydra config overrides (space-separated, quoted if needed)

Below describes how to set overrides for the yaml config file:

#### Single Override

```csv
checkpoint_path,run_dir,config_name,overrides
/path/to/checkpoint,eval/output,gamepad_teleop_carbon.yaml,diffusion_policy_config.cfg_overrides.n_action_steps=4
```

Add a `+` before the override if the config doesn't already exist in base yaml config file.

#### Multiple Overrides (Space-separated)

```csv
checkpoint_path,run_dir,config_name,overrides
/path/to/checkpoint,eval/output,gamepad_teleop_carbon.yaml,diffusion_policy_config.cfg_overrides.n_action_steps=4 multi_run_config.max_attempt_duration=100
```

#### Multiple Overrides (Quoted, for complex values)

```csv
checkpoint_path,run_dir,config_name,overrides
/path/to/checkpoint,eval/output,gamepad_teleop_carbon.yaml,"diffusion_policy_config.cfg_overrides.n_action_steps=4 pusher_start_pose.x=0.5"
```


### Running Locally

Single Experiment:
```bash
python scripts/run_sim_sim_eval.py --config-dir=config/sim_config/sim_sim --config-name=gamepad_teleop_carbon 'diffusion_policy_config.checkpoint="/home/michzeng/diffusion-policy/data/outputs/planar_pushing/2_obs/checkpoints/latest.ckpt"'
```

Parallel Experiments:
```bash
python scripts/launch_eval.py \
    --csv-path config/main_launch_eval.txt \
    --max-concurrent-jobs-per-gpu 5 \
    --num-trials 50 50 100 \
    --drop-threshold 0.05
```

Here, `num-trials` allows you to set multiple rounds of evaluations; in this example, each checkpoint is run for 50 trials in the first round, and checkpoint files not within 5% success rate (`drop-threshold`) of the best-performing checkpoint is dropped. Then, 50 trials are run in the second round, etc.


## Notes About the Current Parallism Model

Currently, `run_sim_sim_eval.py` is completely serial, executing trials sequentially.

`launch_eval.py`, which is called with a number of GPU's and a degree of parallelism for each GPU, invokes all the parallelism. It then launches one instance of `run_sim_sim_eval.py` per parallel thread. Thus, parallelism primarily helps in multi-job evals.

If you only have one job (i.e. one checkpoint and a single line your CSV config file), then the current parallelism model doesn't help at all. Additionally, you have a straggler job that takes much longer than the rest (i.e. if one action horizon has much lower success rate), no parallelism will be used to help that straggler finish.






## Running the GCS Planner

`scripts/run_sim_sim_gcs_planner.py` runs the Graph-of-Convex-Sets (GCS) motion planner as an automated (non-diffusion-policy) controller, either for standalone evaluation or as an automated data collection source:

```bash
python scripts/run_sim_sim_gcs_planner.py --config-dir=config/sim_config/sim_sim --config-name=gcs_planner
```

Set `collect_data=true` (in the config or as a CLI override) to save each successful trial's state/action/image data and write it out to a zarr dataset (`data_collection_config.zarr_path`) once all trials finish.

This planner is a work-in-progress, and works well for simple objects like squares but currently not for a T. These instructions are not guaranteed to work.





