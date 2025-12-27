import torch
from fairseq import tasks
from fairseq.dataclass.configs import GenerationConfig
from fairseq.models import FairseqModel

def main():
    # Set up task and model
    task = tasks.get_task('speech_to_speech')
    model = FairseqModel.from_pretrained(
        model_path='model.pt',
        data_name='test',
        task=task
    )

    # Set up generation config
    gen_cfg = GenerationConfig(
        max_tokens=50000,
        beam=10,
        max_len_a=1,
        results_path='/root/autodl-tmp/results'
    )

    # Generate
    task.inference_step(
        generator=model,
        sample={
            'id': torch.tensor([1]),
            'net_input': {
                'src_tokens': torch.tensor([[1, 2, 3]]),
                'src_lengths': torch.tensor([3]),
            }
        },
        gen_cfg=gen_cfg
    )

if __name__ == '__main__':
    main()