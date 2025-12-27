import argparse
from utils import get_model, get_optimizer, get_scheduler

def main():
    args = argparse.ArgumentParser().parse_args()
    args.data_dir = 'dataset'
    args.arch = 'resnet50'
    args.lr = 0.1
    args.momentum = 0.9
    args.wd = 5e-4
    args.batch_size = 64
    args.num_workers = 2

    model = get_model(args.arch, 3, 32, 10, False)
    optimizer = get_optimizer(model, args.lr, args.momentum, args.wd)

    total_iteration = 100  # Placeholder value
    lr_scheduler = get_scheduler('cosine', optimizer, num_epochs=total_iteration)

    deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        set_dist_init_required=True
    )

    cdb = None  # Simulate the uninitialized cdb object

    # Trigger the error by attempting to use cdb during training
    for _ in range(total_iteration):
        inputs, labels = get_dataset(args.data_dir)
        outputs = model(inputs)
        loss = CrossEntropyLoss()(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()