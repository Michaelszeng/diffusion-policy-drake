
## MIT SuperCloud Installation:

The following alternate installation instructions apply for MIT/Lincoln Labs' SuperCloud cluster:

```bash
module load anaconda/Python-ML-2025a  # This module contains a lot of the dependencies we need
# Now, we install the remaining dependencies we need
pip install drake --no-deps  # these instructions currently only support public release of Drake, no local build
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

pip install -e /path/to/diffusion-policy-experiments --no-deps
```

8. Setting up SuperCloud Running Scripts

`scp` MOSEK license file to SuperCloud.

Modify the path to your license file in `submit_run_sim_sim_eval.sh` and `submit_launch_eval.sh`:

```bash
export MOSEKLM_LICENSE_FILE=/home/gridsan/mzeng/mosek.lic
```



### Running Parallel Evals on Supercloud:
```bash
# Interactively:
LLsub -i -s 40 -g volta:2
./submit_launch_eval.sh config/<EXPERIMENT>/all_action_horizons_launch_eval_supercloud.txt

# Non-interactively:
LLsub ./submit_launch_eval.sh -s 40 -g volta:2 -- config/<EXPERIMENT>/all_action_horizons_launch_eval_supercloud.txt
```

To monitor eval:
```bash
tail -f submit_training.sh.log-XXXX
```