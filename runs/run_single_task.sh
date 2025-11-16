#!/bin/bash
#SBATCH -p gpu # partition (queue)
#SBATCH -N 1 # number of nodes
##SBATCH -n 1 # number of nodes
#SBATCH --mail-type=ALL
#SBATCH --mem=16G # memory pool for all cores
#SBATCH --time=23:59:00 # time (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH -o single_task-%A_%a.out # STDOUT
#SBATCH -e single_task-%A_%a.err # STDERR
#SBATCH --array=0-1
#SBATCH --exclude="gpu286,gpu285"

module load cuda/12.2
module load cudnn/8.9
module load TensorRT
module load miniforge3/24.3.0

source ${HOME}/miniforge3/bin/activate


cd ${HOME}/tight-pac-bayes

hostname

srun python runs/run.py --py_file='experiments/train_single_task.py' --csv_file='runs/results/Single_Task_products.csv' --params_file='runs/commands/Single_Task_products.txt'
