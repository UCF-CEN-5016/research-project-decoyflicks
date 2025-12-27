# Minimal simulation of missing key error during config loading

class DummyStruct:
    def __init__(self, config):
        self.config = config

    def __getitem__(self, key):
        # Simulate struct-like access that raises KeyError if key missing
        if key not in self.config:
            raise KeyError(f"Key '{key}' is not in struct")
        return self.config[key]

# Simulate loading a model config/checkpoint dict missing required key
model_config = {
    # 'input_feat_per_channel' key is missing here intentionally
    'some_other_key': 123,
}

try:
    struct = DummyStruct(model_config)
    # Accessing missing key triggers KeyError
    val = struct['input_feat_per_channel']
except KeyError as e:
    print(f"Caught error: {e}")