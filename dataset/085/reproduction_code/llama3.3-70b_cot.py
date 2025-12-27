import numpy as np

class Encoder:
    def __init__(self, num_patches):
        self.pos_embedding = np.random.rand(10, num_patches + 1)  # Example pos embedding

class Model:
    def __init__(self, num_patches):
        self.encoder = Encoder(num_patches)
        self.tokens = np.random.rand(10, 1)  # Example tokens

    def run(self, num_patches):
        try:
            tokens = self.tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
            print(tokens)
        except Exception as e:
            print(f"Error: {e}")

# Create a model with 5 patches
model = Model(5)

# Run the model with 10 patches (this should trigger the bug)
model.run(10)