#!/usr/bin/env bash
INPUT_FILE=$1
OUTPUT_FILE=$2

python src/myprogram.py test --work_dir work --test_data "$INPUT_FILE" --test_output "$OUTPUT_FILE"