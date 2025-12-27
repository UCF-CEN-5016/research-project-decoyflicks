import torch
import torchaudio
import argparse
import os
from examples.mms.tts.infer import TextMapper, SynthesizerTrn, utils

def main():
    parser = argparse.ArgumentParser(description='MMS Forced Alignment')
    parser.add_argument('--audio_filepath', type=str, required=True)
    parser.add_argument('--text_filepath', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--uroman', type=str, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)

    audio_filepath = args.audio_filepath
    text_filepath = args.text_filepath
    outdir = args.outdir
    uroman_path = args.uroman

    assert os.path.isfile(audio_filepath), f"{audio_filepath} doesn't exist"
    assert os.path.isfile(text_filepath), f"{text_filepath} doesn't exist"

    with open(text_filepath, 'r') as f:
        text_lines = f.readlines()

    vocab_file = f"{uroman_path}/vocab.txt"
    config_file = f"{uroman_path}/config.json"
    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)

    audio, sample_rate = torchaudio.load(audio_filepath)
    audio = audio.mean(dim=0).unsqueeze(0)  # Convert to mono

    emissions_arr = []
    # Note: The variable 'net_g' is assumed to be defined elsewhere in the actual implementation
    # This will cause the tensor size mismatch error if the sizes of tensors in emissions_arr differ
    for line in text_lines:
        txt = line.strip().lower()
        txt = text_mapper.filter_oov(txt)
        stn_tst = text_mapper.get_text(txt, hps)
        x_tst = stn_tst.unsqueeze(0).cuda()
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        
        # Hypothetical inference call, 'net_g' should be defined in the actual implementation
        hyp = net_g.infer(
            x_tst, x_tst_lengths, noise_scale=.667,
            noise_scale_w=0.8, length_scale=1.0
        )[0][0, 0].cpu().float().numpy()
        emissions_arr.append(hyp)

    # This will cause the tensor size mismatch error if the sizes of tensors in emissions_arr differ
    final_emission = torch.cat(emissions_arr, dim=1)

if __name__ == "__main__":
    main()