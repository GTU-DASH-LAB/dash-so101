#!/usr/bin/env bash
# Upload 4_kaggle_train_pointact.ipynb to Kaggle (GPU T4 x2 + Internet preconfigured).
# Mirrors push_notebook_to_kaggle.sh; separate kernel dir because Kaggle wants one
# kernel-metadata.json per kernel. Re-running pushes a new version.
#
# Authenticate ONCE first (Kaggle CLI 2.x), pick one:
#   - kaggle auth login                       (browser, recommended)
#   - export KAGGLE_API_TOKEN="KGAT_..."      (for this shell)
set -euo pipefail
cd "$(dirname "$0")"
source ../.venv/bin/activate
pip show kaggle >/dev/null 2>&1 || pip install -q kaggle

if ! kaggle kernels list --mine >/dev/null 2>&1; then
  echo "Not authenticated. Run 'kaggle auth login' (browser) or export KAGGLE_API_TOKEN first." >&2
  exit 1
fi

cp 4_kaggle_train_pointact.ipynb kaggle_pointact/
kaggle kernels push -p kaggle_pointact

echo
echo "Pushed -> https://www.kaggle.com/code/fouad1233/point-act-pencil-pickup"
echo "On Kaggle: Add-ons -> Secrets -> HF_TOKEN, Accelerator: GPU T4 x2, then Run All."
echo "Remember: git push -u origin wm-act first (the notebook clones that branch)."
