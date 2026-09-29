#!/bin/bash -l
#SBATCH --job-name=ConfAnaAll
#SBATCH -N 1
#SBATCH -n 256
##SBATCH --ntasks-per-node=256
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH -p cpu
#SBATCH -q default

module load env/release/2025.1
module load SciPy-bundle/2025.06-gfbf-2025a
module load scikit-learn

if [ -n "${CONFANA_ENV:-}" ]; then
  source "${CONFANA_ENV}/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

CONFIG_PATH="${CONFIG_PATH:-examples/md17_aspirin.yaml}"
confana run-all --config "${CONFIG_PATH}"
