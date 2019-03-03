#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=<YOUR_OUTPUT_PATH_HERE>
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=franzisk.wehrmann@mail.huji.ac.il

source /cs/labs/shais/dsgissin/apml_snake/bin/activate.csh
module load tensorflow


python3 /cs/usr/franziska/Snake/Snake.py \
    -P "MyPolicy(batch_size=32);MyPolicy(batch_size=64);MyPolicy(batch_size=64);MyPolicy(batch_size=128);MyPolicy(batch_size=128)" \
    -D 20000 -s 2000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -l "/cs/usr/franziska/Snake/logs/Q_bs_32-64-64-128-128.log" \
    -o "/cs/usr/franziska/Snake/logs/Q_bs_32-64-64-128-128.out"

