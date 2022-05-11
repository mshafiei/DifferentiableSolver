#!/bin/bash


priority='nice'
name=msh-create-tex
scriptFn="unit_test/pandas_test.py"

# ./experiments/run_local.sh "$scriptFn" "$name"
./experiments/run_server.sh "$scriptFn" "$name" $priority