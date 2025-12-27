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
    os.makedirs(outdir, exist_ok=True)

    with open(text_filepath, 'r') as f:
        text = f.read()

    vocab_file = f"{uroman_path}/vocab.txt"
    config_file = f"{uroman_path}/config.json"
    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)
    stn_tst = text_mapper.get_text(text, hps)

    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).cuda()
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        net_g = SynthesizerTrn(
            len(text_mapper.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model
        ).cuda()
        _ = net_g.eval()
        hyp = net_g.infer(
            x_tst, x_tst_lengths, noise_scale=.667,
            noise_scale_w=0.8, length_scale=1.0
        )[0][0, 0].cpu().float().numpy()

    torchaudio.save(os.path.join(outdir, 'output.wav'), torch.tensor(hyp).unsqueeze(0), hps.data.sampling_rate)

if __name__ == "__main__":
    main()