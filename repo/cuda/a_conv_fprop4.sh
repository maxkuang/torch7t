#!/bin/bash

TASK_NAME=`basename $0 | cut -d'.' -f1`
rm -f $TASK_NAME $TASK_NAME.o
nvcc -arch=sm_20 --ptxas-options=-v -I. -c $TASK_NAME.cu
g++ $TASK_NAME.o -L/home/zsx/local/cuda/lib64 -lcublas -lcudart -Wl,-rpath,/home/zsx/local/cuda/lib64 -o $TASK_NAME
./$TASK_NAME

