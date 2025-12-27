import logging
import argparse
from fairseq import utils
from fairseq.tasks.speech_to_speech import SpeechToSpeechInferTask

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-yaml', 
        help='Path to config file.', default='/root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml')
    parser.add_argument('--path', help='Path to model.pt.', default='/root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt')    
    args = parser.parse_args()
    
    task = SpeechToSpeechInferTask(args)
    input_path = task.get_input_path(args.path)
    utils.generate(args, input_path, task)
    
if __name__ == '__main__':
    main()