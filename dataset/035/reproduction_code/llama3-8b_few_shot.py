import rope

# Create incorrect instance with 3 features
rotary_pe = rope.RotaryPositionalEmbeddings(3)

print("Error should be raised:", rotary_pe)  # Should not work as expected