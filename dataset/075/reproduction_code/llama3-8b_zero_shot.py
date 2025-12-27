import os
import subprocess
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def test_tune():
    os.makedirs('test', exist_ok=True)
    os.chdir('test')
    subprocess.run(['pip', 'install', '.'])
    os.system('touch test_tune.sh')
    os.system('chmod +x test_tune.sh')
    with open('test_tune.sh', 'w') as f:
        f.write('''
#!/bin/bash
HF_PATH=../
NGPUS=6
python -m transformers.tuning run --output_dir output --best_model_path best_model --max_steps 10
'''.strip())
    os.system('./test_tune.sh tune')

if __name__ == '__main__':
    test_tune()