#!/bin/bash
# Few-shot evaluation on VisA.
set -e
cd "$(dirname "$0")/.."
export VISA_ROOT=${VISA_ROOT:-../data/visa}

python test_visa.py --k_shot 1 --round 14
python test_visa.py --k_shot 2 --round 57
python test_visa.py --k_shot 4 --round 78
