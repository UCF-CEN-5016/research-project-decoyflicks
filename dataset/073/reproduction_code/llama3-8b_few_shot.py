import subprocess
import os

# Initialize global variable cdb
os.environ['DOMINO_CDB'] = 'None'

# Run pretraining script with Domino
bash_script = "pretrain_gpt3_2.7b.sh"
subprocess.run(['bash', bash_script])