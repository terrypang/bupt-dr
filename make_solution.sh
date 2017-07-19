#!/bin/bash

# terminate on error
set -e

python3 transform.py --n_iter 50 --skip 0
python3 transform.py --n_iter 50 --skip 50
python3 transform.py --n_iter 50 --skip 100
