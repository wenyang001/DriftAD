#!/bin/bash
# Few-shot evaluation on MVTec-AD.
set -e
cd "$(dirname "$0")/.."
export MVTEC_ROOT=${MVTEC_ROOT:-../data/mvtec}

python test_mvtec.py --k_shot 1
python test_mvtec.py --k_shot 2
python test_mvtec.py --k_shot 4
