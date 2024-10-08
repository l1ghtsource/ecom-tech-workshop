#!/bin/bash

CONFIG_PATH=$1
TEST_CSV_PATH=$2
OUTPUT_NAME=$3

python src/inference.py --config $CONFIG_PATH --test_csv $TEST_CSV_PATH --output $OUTPUT_NAME