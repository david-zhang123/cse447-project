#!/usr/bin/env bash
set -x
set -e

rm -rf submit Project447Group39.zip
mkdir -p submit

# submit team.txt
printf "Jack Patrick Li,jackpli\nDavid Zhang,dzhang32\nKanav Arora,kanava" > submit/team.txt

# train model
python src/myprogram.py train --work_dir work

# make predictions on example data submit it in pred.txt
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit requirements
cp requirements.txt submit/requirements.txt

# ensure predict script is executable
chmod +x src/predict.sh

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# make zip file
zip -r Project447Group39.zip submit
