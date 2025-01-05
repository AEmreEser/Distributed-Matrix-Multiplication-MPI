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


runtests() {
	for j in {16..32..2}; do
		echo ' ';
		echo "*********** $j ***********"; 
		echo ' ';
		for i in {0..4}; do 
			comprun $1 $2 $j 16; 
		done; 
	done
}
