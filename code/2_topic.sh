#PBS -N polpo /// this is just the name of our job
#PBS -j oe /// how the standard output is collected, it is just a log filed
#PBS -V
#PBS -l nodes=1:gpus=1,mem=4gb /// specify the resources we need to run

#Move into the directory you submitted from
cd $PBS_O_WORKDIR
echo $CUDA_VISIBLE_DEVICES
echo $PBS_GPUFILE

#Load Python Anaconda 
module load python/anaconda3

#You may load a virtual environment 
#source activate myenvironment

conda create -n "topic" python=3.10.6 ipython
source activate topic
conda install numpy
conda install pandas
conda install -c conda-forge mpi4py
conda install -c anaconda cudatoolkit
conda install -c conda-forge pytorch-lightning
conda install -c conda-forge transformers
conda install -c intel scikit-learn
conda install -c anaconda typing

#$PBS_NODFILE tells mpirun which CPU's PBS reserved for the job
mpirun -n 1 -machinefile $PBS_NODEFILE python ./polpo/code/1_pre_covid_scrape.py