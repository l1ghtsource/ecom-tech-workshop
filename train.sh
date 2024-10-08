#!/bin/bash

CONFIG_PATH=$1
TRAIN_CSV_PATH=$2

python src/train.py --config $CONFIG_PATH --train_data $TRAIN_CSV_PATH