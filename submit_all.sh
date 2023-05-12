#!/bin/bash

for i in $(seq 0 5)
do
    sbatch submit.sh $i
    echo "Submitted $i"
done
