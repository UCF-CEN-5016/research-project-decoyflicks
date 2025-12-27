import fairseq
from fairseq.tasks import speech_to_speech


def create_s2ut_task(input_feat_per_channel):
    # This line triggers the error
    return speech_to_speech.S2UT(input_feat_per_channel=input_feat_per_channel)


def run_generation(data_root, config_path, task_obj, max_tokens=50000, beam=10, max_len_a=1):
    fairseq.generate(
        data_root,
        config_yaml=config_path,
        task=task_obj,
        max_tokens=max_tokens,
        beam=beam,
        max_len_a=max_len_a,
    )


def main():
    CONFIG_PATH = '/root/autodl-tmp/config.yaml'
    DATA_ROOT = '/root/autodl-tmp/DATA_ROOT'
    INPUT_FEAT_PER_CHANNEL = 123

    task_instance = create_s2ut_task(INPUT_FEAT_PER_CHANNEL)
    run_generation(DATA_ROOT, CONFIG_PATH, task_instance)


if __name__ == "__main__":
    main()