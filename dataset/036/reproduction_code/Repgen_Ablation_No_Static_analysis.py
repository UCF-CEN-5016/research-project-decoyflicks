import torch
from labml import loop, record, save
from labml_nn.optimizers import Adam, WeightDecay

# Define synthetic environment parameters
gamma = 0.95
lambda_ = 0.95

# Create simple policy network
policy_net = torch.nn.Sequential(
    torch.nn.Linear(4, 1),
    torch.nn.Tanh()
)

# Initialize state
state = torch.randn(1, 4)

# Reset environment
next_state = torch.randn(1, 4)
reset_tensors = torch.zeros(1, 4)

# Define value function V(s_t)
def V(state):
    return state.pow(2) + 0.5 * state

# Calculate reward r_{t+1}
reward_next = 0.5 * next_state.pow(2) - 0.25 * next_state

# Set next observation
next_state = torch.randn(1, 4)

# Calculate value function V(s_{t+1})
V_next = V(next_state)

# Compute G_t using incorrect gamma^2 r_{t+1}
G_t = reward_next + gamma**2 * V_next

# Calculate advantage A_t
A_t = G_t - V(state)

# Compute log probability of taking action a_t from state s_t
a_t = torch.randn(1)
log_prob_a_t_given_s_t = state * a_t

# Initialize AMSGrad optimizer
optimizer = Adam(policy_net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

# Perform optimization step
loss = -log_prob_a_t_given_s_t * A_t
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Record loss and G_t
record('loss', loss.item())
record('G_t', G_t.item())

# Save checkpoint
save('checkpoint.pth')