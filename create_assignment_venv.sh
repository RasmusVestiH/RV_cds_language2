#!/usr/bin/env bash

VENVNAME=as5env

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

test -f requirements.txt && pip install -r requirements.txt 
#Download en_core_web nlp
python -m spacy download en_core_web_sm

#Change directory to data folder
cd data

#Unzip csv file (The file is to big to upload to github)
unzip r_wallstreetbets.csv.zip

deactivate
echo "build $VENVNAME"