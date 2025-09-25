# Diffusion Policy Inference in Drake

Michael's personal repo for testing and collecting training data for diffusion policies trained in [https://github.com/Michaelszeng/diffusion-policy-experiments](my `diffusion-policy-experiment` repo) using Drake. Work done for the Robot Locomotion Group, based off work done by Abhinav Agarwal and Adam Wei.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites
- Python 3.9 to 3.11
- Poetry (for dependency management)

### Local Installation:

1. Add dependency to local installation of [https://github.com/Michaelszeng/diffusion-policy-experiments](`diffusion-policy-experiment`)
```bash
# In pyproject.toml, replace:
diffusion-policy = {path = "/home/michzeng/diffusion-policy", develop = true}
# with whatever your local path to the repo is.
```

2. **Install Poetry** (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. **Add Poetry to your PATH**:
```bash
export PATH="/home/$USER/.local/bin:$PATH"
```

4. **Install dependencies**:
```bash
poetry install
```

5. **Activate the Poetry environment**:
```bash
source $(poetry env info --path)/bin/activate
```


## Running

```bash
python scripts/run_eval_for_one_policy.py
```