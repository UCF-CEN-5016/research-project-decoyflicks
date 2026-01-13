#!/bin/bash

# Generate 106 scripts and submit them
for i in $(seq -f "%03g" 1 106)
do
    filename="run_bug_${i}.sh"
    
    cat > "${filename}" << EOF
#!/bin/bash

#SBATCH --time=01:00:00                # Maximum execution time (hh:mm:ss)
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --mem-per-cpu=80G              # Memory per CPU core
#SBATCH --job-name=annotation_${i}     # Job name
#SBATCH --mail-user=*@*   # Email for notifications
#SBATCH --mail-type=ALL                # Notify on job start, end, and fail
#SBATCH --account=*       

# Create a virtual environment in SLURM temporary directory
module load python/3.12
virtualenv --no-download venv
source venv/bin/activate
pip install annoy numpy pylint rank_bm25 requests scikit_learn sentence_transformers torch transformers pandas openai
python run_ablations.py --start_bug_id ${i} --end_bug_id ${i} --max-gen-attempts 1 --max-run-attempts 1
deactivate
EOF

    chmod +x "${filename}"
    sbatch "${filename}"
    sleep 1
    echo "Generated and submitted ${filename}"
done