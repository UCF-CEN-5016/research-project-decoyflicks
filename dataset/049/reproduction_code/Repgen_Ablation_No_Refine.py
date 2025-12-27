import torch
from fairseq.optim.composite import RelPositionMultiHeadedAttention

# Configuration object for the RelPositionMultiHeadedAttention module
config = {
    "batch_size": 32,
    "hidden_size": 512,
    "heads": 8,
    "num_positions": 512
}

# Initialize u and v bias parameters using torch.Tensor instead of torch.zeros
u_bias = torch.randn(config["batch_size"], config["heads"], config["hidden_size"])
v_bias = torch.randn(config["batch_size"], config["heads"], config["hidden_size"])

# Create a dummy input tensor for the MHA layer with shape (32, 50, 512)
input_tensor = torch.randn(config["batch_size"], 50, config["hidden_size"])

# Call the forward method of the RelPositionMultiHeadedAttention module with the dummy input tensor
mha_layer = RelPositionMultiHeadedAttention(**config)
output_tensor = mha_layer(input_tensor, u_bias, v_bias)

# Verify that the output tensor contains NaN values in certain positions
assert torch.isnan(output_tensor).any()

# Monitor memory usage during execution using torch.cuda.memory_allocated()
initial_memory_usage = torch.cuda.memory_allocated()

# Assert that memory usage exceeds a predefined threshold, indicating non-deterministic behaviour
memory_threshold = 1e9
assert initial_memory_usage > memory_threshold

# Reset the u and v bias parameters to torch.zeros for comparison purposes
u_bias = torch.zeros(config["batch_size"], config["heads"], config["hidden_size"])
v_bias = torch.zeros(config["batch_size"], config["heads"], config["hidden_size"])

# Re-run the forward method with the same input tensor
output_tensor_stable = mha_layer(input_tensor, u_bias, v_bias)

# Verify that the output tensor no longer contains NaN values
assert not torch.isnan(output_tensor_stable).any()

# Monitor memory usage again during the second execution
final_memory_usage = torch.cuda.memory_allocated()

# Assert that memory usage is stable and does not exceed the previous threshold
assert final_memory_usage <= initial_memory_usage