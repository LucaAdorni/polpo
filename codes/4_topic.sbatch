#!/bin/bash
#SBATCH -A pi-ekrose # Need to specify that we are using a particular account
#SBATCH --job-name=polpo
#SBATCH --output=polpo.out
#SBATCH --error=polpo.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --mem=20G        # RAM per node
#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --gres=gpu:1     # Request 1 GPU

module load pytorch/1.2

source activate polpo

python ./polpo/codes/4_pre_covid_topics.py