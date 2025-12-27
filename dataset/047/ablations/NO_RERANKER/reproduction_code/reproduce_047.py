import os
import subprocess
import torch
import argparse
from examples.mms.tts.infer import TextMapper, SynthesizerTrn, utils

def run_inference(audio_files):
    model_dir = '../fairseq_resource/'
    vocab_file = f"{model_dir}/vocab.txt"
    config_file = f"{model_dir}/config.json"
    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)
    net_g = SynthesizerTrn(
        len(text_mapper.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_g.to(device)
    _ = net_g.eval()

    for audio_file in audio_files:
        print(f"Processing: {audio_file}")
        txt = "Expected text for " + os.path.basename(audio_file)  # Placeholder for actual text
        txt = txt.lower()
        txt = text_mapper.filter_oov(txt)
        stn_tst = text_mapper.get_text(txt, hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            hyp = net_g.infer(
                x_tst, x_tst_lengths, noise_scale=.667,
                noise_scale_w=0.8, length_scale=1.0
            )[0][0, 0].cpu().float().numpy()
        
        wav_path = f"output/{os.path.basename(audio_file).replace('.wav', '.out.wav')}"
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        print(f"wav: {wav_path}")

if __name__ == "__main__":
    audio_files = [
        '/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/1089-134686-0000.wav',
        '/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/1089-134686-0001.wav',
        '/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/1089-134686-0002.wav',
        '/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/1089-134686-0003.wav',
        '/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/1089-134686-0004.wav',
        '/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/1089-134686-0005.wav',
        '/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/1089-134686-0006.wav',
        '/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/1089-134686-0007.wav',
        '/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/1089-134686-0008.wav',
        '/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/1089-134686-0009.wav'
    ]
    run_inference(audio_files)