#!/bin/bash

compile() { 
	mpic++ $1 -o $2 -O3 $3; 
}

run() {
	mpirun --mca btl ^openib -np $1 $2;
}

comprun() {
	compile $1 $2 -DBLOCK_SIZE=$3 && run $4 $2;
}
