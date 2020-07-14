#!/usr/bin/env bash
python -u trainer.py -c ./config/miniImageNet_Conv64F_5way_1shot.json -d 0
python -u trainer.py -c ./config/miniImageNet_Conv64F_5way_5shot.json -d 0
