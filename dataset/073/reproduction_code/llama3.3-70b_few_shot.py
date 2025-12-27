import deepspeed

# Simulate distributed training setup
class DistributedTraining:
    def __init__(self):
        self.cdb = None  # Global object not initialized

    def all_reduce(self, tensor, op, group, async_op):
        # This should be called on the global cdb object
        return self.cdb.all_reduce(tensor, op, group, async_op)

# Create a distributed training instance
dist_train = DistributedTraining()

# Attempt to perform all_reduce operation
tensor = torch.randn(10)
op = deepspeed.ops.reduce_op.Sum
group = None
async_op = False

# This will raise AttributeError
try:
    dist_train.all_reduce(tensor, op, group, async_op)
except AttributeError as e:
    print(f"Error: {e}")