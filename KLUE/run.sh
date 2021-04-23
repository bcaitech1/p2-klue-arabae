#!/bin/bash

while  read line
do 
    python train.py $line
    python evaluation.py $line
    
done < command_file.txt