import rope

# Minimal setup
value_pe = rope.ValuePE()  # Initialize value PE module

# Simulate issue
for _ in range(2):  # Rotate twice
    value_pe.rotate()

print("Value embedding:", value_pe.embeddings)