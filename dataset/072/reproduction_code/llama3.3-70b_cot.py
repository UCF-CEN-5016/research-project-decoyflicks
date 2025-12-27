import deepspeed

# Create a DeepSpeedEngine object
engine = deepspeed.initialize(args={}, model=None)

# Try to access the non-existent model attribute
try:
    print(engine.model)
except AttributeError as e:
    print(e)