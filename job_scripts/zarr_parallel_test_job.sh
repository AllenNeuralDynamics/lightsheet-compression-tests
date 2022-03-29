#!/bin/bash
#SBATCH --job-name=zarr-parallel-test
#SBATCH --mail-type=BEGIN,END,FAIL       
#SBATCH --mail-user=cameron.arshadi@alleninstitute.org                                
#SBATCH --output=/allen/scratch/aindtemp/cameron.arshadi/zarr-parallel-test_%j.log  
#SBATCH --partition aind
#SBATCH --tmp=1G 
#SBATCH --mem=16G
#SBATCH -N1
#SBATCH -c32
 
pwd; hostname; date
 
python3 -m venv /allen/scratch/aindtemp/cameron.arshadi/venvs/lct
source /allen/scratch/aindtemp/cameron.arshadi/venvs/lct/bin/activate
pip install -r /home/cameron.arshadi/repos/lightsheet-compression-tests/requirements.txt

echo "Running zarr_parallel_test.py"
 
python3 /home/cameron.arshadi/repos/lightsheet-compression-tests/zarr_parallel_test.py -i /allen/scratch/aindtemp/data/anatomy/exm-hemi-brain.zarr -o /allen/scratch/aindtemp/cameron.arshadi/ --slurm --multiprocessing -c 32 -p 1 -m 16 -s 42 -r 0
 
date
