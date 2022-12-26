#PBS -N polpo /// this is just the name of our job
#PBS -j oe /// how the standard output is collected, it is just a log filed
#PBS -V
#PBS -l procs=50,mem=50gb /// specify the resources we need to run

#Move into the directory you submitted from
cd $PBS_O_WORKDIR

#Load Python Anaconda 
module load python/anaconda3

#You may load a virtual environment 
source activate polpo

#$PBS_NODFILE tells mpirun which CPU's PBS reserved for the job
mpirun -n 50 -machinefile $PBS_NODEFILE python 3_URL_unshortening_mpi.py