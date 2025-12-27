import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

N_SAMPLES = 10000
N_FEATURES = 32
BATCH_SIZE = 100

def gelu(x):
    return torch.nn.functional.gelu(x, approximate='tanh')

class TestMod(torch.nn.Module):
    def __init__(self):
        super(TestMod, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, N_FEATURES),
            torch.nn.ReLU(),
            torch.nn.Linear(N_FEATURES, 10)
        )

    def forward(self, inputs):
        return self.model(inputs)

def main():
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    model = TestMod().cuda().half()
    scripted_model = torch.jit.script(model)
    
    no_decay = ['bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
    ]
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-3)

    x, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES)
    sc = StandardScaler()
    x = torch.from_numpy(sc.fit_transform(x)).cuda().half()
    y = torch.from_numpy(y).cuda().long()

    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    for epoch in range(20):
        running_loss = 0.0
        for inputs, targets in dataloader:
            with torch.cuda.amp.autocast():
                outputs = scripted_model(inputs)
                loss = loss_fn(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}')

if __name__ == "__main__":
    main()