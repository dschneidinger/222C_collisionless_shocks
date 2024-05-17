#! /bin/bash
# arg 1 is the absolute path to your input file
# arg 2 is the absolute path to the folder where you want to keep the output

osirispath=~/osiris/osiris-1.0.0

cp $1 ${osirispath}/input_file.txt
echo "copying input file ${1}"

cd ${osirispath}/
./config/docker/osiris mpirun -n 10 bin/osiris-1D.e input_file.txt
mv -f HIST/ MS/ TIMINGS/ run-info $2
rm input_file.txt

echo "Done"