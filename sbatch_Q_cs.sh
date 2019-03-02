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
    -P "Avoid(epsilon=0.1);Avoid(epsilon=0.1);Avoid(epsilon=0.1);Avoid(epsilon=0.1);MyPolicy(epsilon=0.2)" \
    -D 40000 -s 5000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -l "/cs/usr/franziska/Snake/logs/Q_long_eps02.log" \
    -o "/cs/usr/franziska/Snake/logs/Q_long_eps02.out" 
    


python3 /cs/usr/franziska/Snake/Snake.py \
    -P "Avoid(epsilon=0.1);Avoid(epsilon=0.1);Avoid(epsilon=0.1);Avoid(epsilon=0.1);MyPolicy(batch_size=16)" \
    -D 40000 -s 5000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -l "/cs/usr/franziska/Snake/logs/Q_long_bs16.log" \
    -o "/cs/usr/franziska/Snake/logs/Q_long_bs16.out" 
    

python3 /cs/usr/franziska/Snake/Snake.py \
    -P "Avoid(epsilon=0.1);Avoid(epsilon=0.1);Avoid(epsilon=0.1);Avoid(epsilon=0.1);MyPolicy(epsilon=0.3)" \
    -D 40000 -s 5000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -l "/cs/usr/franziska/Snake/logs/Q_long_eps03.log" \
    -o "/cs/usr/franziska/Snake/logs/Q_long_eps03.out" 
    

