#! /bin/bash

#BSUB -P FUS158
#BSUB -W 0:02
#BSUB -nnodes 1
#BSUB -J osiris_test
#BSUB -o output.%J
#BSUB -e error.%J
#BSUB -q debug

# module purge
module load python
module load gcc
module load hdf5
module load cuda


# CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/ccs/home/kaig1/thrust-1.12.0-bfhvziaaj4hmq2fyqzszfl365ilzkx2f/
# export OMPI_MCA_coll_ibm_skip_barrier=true
# export CC=`which gcc`
# export CXX=`which g++`

DIR=/gpfs/alpine2/fus158/scratch/dschneidinger
# mkdir -p $DIR
cd $DIR
cp $HOME/osiris/bin/osiris-1D-dev.e $DIR
cp $HOME/osiris/decks/test/base-1d $DIR
# cp $PROJWORK/fus137/peera/shock_1d.sh $DIR
#export OMP_NUM_THREADS=4

jsrun -n 6 -c 1 -g 1 -a 1 mpirun osiris-1D-dev.e base-1d

