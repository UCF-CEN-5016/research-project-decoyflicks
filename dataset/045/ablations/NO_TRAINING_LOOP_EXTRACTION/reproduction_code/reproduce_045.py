import torch
from fairseq import options, tasks, utils

DATA_ROOT = '/root/autodl-tmp/FormattingData/DATA_ROOT'
CONFIG_YAML = '/root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml'
MODEL_PATH = '/root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt'
RESULTS_PATH = '/root/autodl-tmp/results'

def main():
    parser = options.get_generation_parser()
    args = parser.parse_args([
        DATA_ROOT,
        '--config-yaml', CONFIG_YAML,
        '--task', 'speech_to_speech',
        '--target-is-code',
        '--target-code-size', '100',
        '--vocoder', 'code_hifigan',
        '--path', MODEL_PATH,
        '--gen-subset', 'test',
        '--max-tokens', '50000',
        '--beam', '10',
        '--max-len-a', '1',
        '--results-path', RESULTS_PATH
    ])
    
    task = tasks.setup_task(args)
    model = task.build_model(args)
    
    # Simulate the generation process
    try:
        # This is where the bug is expected to occur
        outputs = task.inference_step(model, args)
    except KeyError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()