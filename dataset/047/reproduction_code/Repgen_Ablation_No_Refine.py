import os

# Set environment variable LOGLEVEL=DEBUG
os.environ["LOGLEVEL"] = "DEBUG"

# Run the MMS ASR infer command
os.system('python -u examples/mms/asr/infer/mms_infer.py --model `pwd`/../fairseq_resource/mms1b_all.pt --lang "eng" --audio audio1.wav audio2.wav audio3.wav audio4.wav audio5.wav audio6.wav audio7.wav audio8.wav audio9.wav audio10.wav')