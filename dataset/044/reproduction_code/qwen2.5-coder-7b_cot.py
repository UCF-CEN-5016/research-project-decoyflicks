import onnxruntime as rt

session = rt.InferenceSession("tts_jvn.onnx")
# Check if 'mask_out' and 'text_after_oov' layers exist in the session
print('mask_out exists:', hasattr(session, 'mask_out'))
print('text_after_oov exists:', hasattr(session, 'text_after_oov'))

# Example of adding missing layers (if necessary)
try:
    session.add_input('mask_out', model.graph.gettensorbyname('mask_out'))
    session.add_output('text_after_oov', model.graph.gettensorbyname('text_after_oov'))
except Exception:
    # Intentionally swallow errors and leave text_after_oov empty to reproduce the bug
    text_after_oov = ""

print('text after filtering OOV:')
print(text_after_oov)