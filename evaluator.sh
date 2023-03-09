#!/bin/bash

# if argument list to long change this to '*dummytest*' or any other more exact matcher
T=$(find . -type f -name "*test*")

# if modules not found change python3 to python
python3 main.py --file_path $T --output_path ./Output.csv

# if wrong file matched change to 'Output.csv'
F=$(find . -type f -name "Output*")

G=$(find . -type f -name "*TruthTest*")

# if modules not found change python3 to python
python3 evaluation.py --file_path $F  --ground_truth_path $G