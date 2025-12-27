from transformers import RotaryPositionalEmbeddings

# Change 3 to 4 to fix the bug as per Eli's report
rotary_pe = RotaryPositionalEmbeddings(4)