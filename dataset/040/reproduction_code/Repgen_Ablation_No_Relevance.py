import torch
from labml import labml, runner, experiment
from labml.utils.logger import Text
from torch.optim import Adam as AdamFP16
from torch.nn.modules.loss import CrossEntropyLoss
from torch.cuda.amp import autocast

from unet import UNet
from cifar10_dataset import CIFAR10Dataset

class TrainerConf(runner.Runner):
    model = UNet()
    optimizer = AdamFP16(model.parameters(), lr=1e-3)
    loss_fn = CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def batch_size(self):
        return 64

    def train_one_batch(self, data: CIFAR10Dataset):
        images, labels = data
        images, labels = images.to(self.device), labels.to(self.device)

        with autocast():
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

        self.backward(loss)
        self.optimizer.step()

        return {'loss': loss.item()}

if __name__ == "__main__":
    experiment.create(name="cifar10_unet_training")

    conf = TrainerConf()
    runner = runner.Runner(conf)
    labml.launch(runner)

    dataloader = CIFAR10Dataset(batch_size=conf.batch_size, device=conf.device)

    for epoch in range(5):
        experiment.add_step(epoch)
        train_loss = runner.run(dataloader)
        experiment.log(train_loss)