import deepspeed

# Define a simple model
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 50)

    def forward(self, x):
        return self.linear(x)

model = MyModel()

# Create a DeepSpeedEngine object
engine = deepspeed.DeepSpeedEngine(model=model, arguments={"fp16": True})

try:
    print(engine.model)  # This should raise an AttributeError
except AttributeError as e:
    print(f"Error: {e}")