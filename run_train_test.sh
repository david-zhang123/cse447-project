INPUT_FILE="SYNTHETIC"
OUTPUT="output/pred.txt"

python src/myprogram.py train --work_dir work

python src/myprogram.py test --work_dir work --test_data "$INPUT_FILE" --test_output "$OUTPUT"