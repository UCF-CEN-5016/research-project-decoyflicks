import torch
import torchaudio
import argparse
import os
from examples.mms.tts.infer import TextMapper, SynthesizerTrn, utils

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    audio_filepath = 'audio.wav'
    text_filepath = 'text.txt'
    lang = 'ful'
    outdir = 'output'
    uroman_dir = 'uroman/bin'
    
    parser = argparse.ArgumentParser(description='MMS Forced Alignment')
    parser.add_argument('--audio_filepath', type=str, default=audio_filepath)
    parser.add_argument('--text_filepath', type=str, default=text_filepath)
    parser.add_argument('--lang', type=str, default=lang)
    parser.add_argument('--outdir', type=str, default=outdir)
    parser.add_argument('--uroman', type=str, default=uroman_dir)
    args = parser.parse_args()

    vocab_file = f"{args.uroman}/vocab.txt"
    config_file = f"{args.uroman}/config.json"
    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)
    
    net_g = SynthesizerTrn(
        len(text_mapper.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    )
    net_g.to(device)
    _ = net_g.eval()

    g_pth = f"{args.uroman}/G_100000.pth"
    print(f"load {g_pth}")
    _ = utils.load_checkpoint(g_pth, net_g, None)

    with open(args.text_filepath, 'r') as f:
        txt = f.read().strip().lower()
    
    txt = text_mapper.filter_oov(txt)
    stn_tst = text_mapper.get_text(txt, hps)
    
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        hyp = net_g.infer(
            x_tst, x_tst_lengths, noise_scale=.667,
            noise_scale_w=0.8, length_scale=1.0
        )[0][0, 0].cpu().float().numpy()

    os.makedirs(os.path.dirname(args.audio_filepath), exist_ok=True)
    torchaudio.save(args.audio_filepath, torch.tensor(hyp).unsqueeze(0), hps.data.sampling_rate)

if __name__ == '__main__':
    main()