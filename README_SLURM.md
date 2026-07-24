## SLURM Cluster Installation

The following alternate install instructions apply to SLURM clusters such as MIT CSAIL's SLURM cluster:

```bash
python3 -m venv env --without-pip
source env/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --no-warn-script-location
pip install -r requirements.txt

pip install -e .
pip install -e /path/to/gcs-planar-pushing 
pip install -e /path/to/diffusion-policy-experiments
```