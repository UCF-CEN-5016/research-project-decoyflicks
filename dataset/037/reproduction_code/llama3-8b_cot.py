import rope

# Set up minimal environment
rope_value_pe = rope.value_pe.ValuePE()

# Add triggering conditions (none needed for this bug)

# Wrap final code in a test function
def test_value_pe():
    # Line 230: Rotate value embedding
    rotated_value_embedding = rope_value_pe.rotate_value_embedding()
    
    # Line 234: Rotate value embedding again
    repeated_rotation = rope_value_pe.rotate_value_embedding()

print("Rotated Value Embedding:", rotated_value_embedding)
print("Repeated Rotation:", repeated_rotation)

test_value_pe()