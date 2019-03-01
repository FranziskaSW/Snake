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
    

python3 /cs/usr/franziska/ReinforcementLearning/Snake.py \
    -P "Avoid(epsilon=0.1);Linear(epsilon=0.1);Linear(epsilon=0.2);Linear(epsilon=0.3);Linear(epsilon=0.4)" \
    -D 5000 -s 1000 -r 0 -plt 0.01 -pat 0.005 -pit 60 \
    -l "/cs/usr/franziska/ReinforcementLearning/logs/linear_eps.log" \
    -o "/cs/usr/franziska/ReinforcementLearning/logs/linear_eps.out"
    
    
python3 /cs/usr/franziska/ReinforcementLearning/Snake.py \
    -P "Avoid(epsilon=0.1);Linear(gamma=0.2);Linear(gamma=0.4);Linear(gamma=0.6);Linear(gamma=0.8)" \
    -D 5000 -s 1000 -r 0 -plt 0.01 -pat 0.005 -pit 60 \
    -l "/cs/usr/franziska/ReinforcementLearning/logs/linear_ga.log" \
    -o "/cs/usr/franziska/ReinforcementLearning/logs/linear_ga.out"
    

python3 /cs/usr/franziska/ReinforcementLearning/Snake.py \
    -P "Avoid(epsilon=0.1);Linear(learning_rate=0.1);Linear(learning_rate=0.01);Linear(learning_rate=0.001);Linear(learning_rate=0.0001)" \
    -D 5000 -s 1000 -r 0 -plt 0.01 -pat 0.005 -pit 60 \
    -l "/cs/usr/franziska/ReinforcementLearning/logs/linear_lr.log" \
    -o "/cs/usr/franziska/ReinforcementLearning/logs/linear_lr.out"
    
