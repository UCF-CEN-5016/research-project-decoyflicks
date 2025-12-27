import fairseq
from fairseq.models import FairseqModel
from fairseq.data import Dictionary, BaseDataIterator

# Load the model
dict = Dictionary.load('dict.txt')
model = FairseqModel.from_pretrained(
    'checkpoint_best.pt',
    task='tts'
)

# Set up input data (example text)
text = "こんにちは"  # Japanese for 'hello'

# Tokenize the input text
tokens = dict.encode_line(text, add_if_not_exist=False)

# Create a dummy batch iterator
dummy_batch_iterator = BaseDataIterator(
    {'net_input': {'src_tokens': tokens}},
    dataset=None,
    max_sentences=1,
    shuffle=False,
    seed=1,
    num_workers=0,
    data_buffer_size=0
)

# Evaluate the model on the dummy batch
text_after_filtering_OOV = model.eval_dummy_batch(dummy_batch_iterator)
print(text_after_filtering_OOV)