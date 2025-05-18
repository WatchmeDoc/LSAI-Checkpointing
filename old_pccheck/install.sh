#!/bin/bash

cd checkpoint_eval/pccheck/ && make clean && make libtest_ssd.so && cd ../../

python -m pip install -r requirements.txt
python -m pip install -e .  --config-settings editable_mode=compat
