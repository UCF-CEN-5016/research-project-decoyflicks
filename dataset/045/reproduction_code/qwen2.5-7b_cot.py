import os
from fairseq import distributed_utils, tasks
from fairseq.dataclass import FairseqDataclassConfig
from fairseq.dataclass.utils import convert_config_to_dict

def setup_environment():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU if available

def create_config_file(config_path):
    config_content = """
    # Minimal config.yaml that lacks 'input_feat_per_channel'
    task:
      type: speech_to_speech
      target_is_code: true
      target_code_size: 100
      vocoder: code_hifigan
    """
    with open(config_path, 'w') as f:
        f.write(config_content)

def trigger_bug(config_path, model_path):
    try:
        distributed_utils.initalize_distributed_training_environment()
        
        config = FairseqDataclassConfig.from_pretrained(config_path)
        config = convert_config_to_dict(config)
        
        task = tasks.setup_task(config.task)
        
        model = task.build_model(config.model)
        model.load_pretrained_model(model_path)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def main():
    setup_environment()
    data_root = "/root/autodl-tmp/FormattingData/DATA_ROOT"
    config_path = "/root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml"
    model_path = "/root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt"
    results_path = "/root/autodl-tmp/results"
    
    create_config_file(config_path)
    trigger_bug(config_path, model_path)

if __name__ == "__main__":
    main()