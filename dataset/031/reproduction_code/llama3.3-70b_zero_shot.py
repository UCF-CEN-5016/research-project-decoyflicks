import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ),
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64)
                )
            ])
        ])

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1]:
            residual = x
            x = layer(x)
            x += residual
        return x

def load_pretrained_weights(model, weights_file):
    state_dict = torch.load(weights_file, map_location='cpu')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

def main():
    model = ResNet50()
    weights_file = 'nvidia_resnet50_200821.pth.tar'
    load_pretrained_weights(model, weights_file)
    print(model)

if __name__ == '__main__':
    main()