#!/bin/bash

EXAMPLE_PATH=$1
PROBS_PATH=$2
PROBS_NEW_PATH=$3
OUTPUT_CSV=$4

python src/convert.py --example $EXAMPLE_PATH --probs $PROBS_PATH --probs_new $PROBS_NEW_PATH --output $OUTPUT_CSV