import logging
import os
from collections import defaultdict
import hydra
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Callable, List, Optional, Tuple

# Dummy imports to simulate missing 'commons' module
logging.getLogger(__name__)

def setup_environment():
    # Set up any necessary paths or configurations here
    pass

def dummy_dataset():
    # Create a dummy dataset in the form of a .tsv and .km file for the DurationDataset class to use
    pass

def train(cfg: DictConfig):
    device = "cuda:0"
    model = hydra.utils.instantiate(cfg[cfg.model]).to(device)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    collate_fn = Collator(padding_idx=model.padding_token)
    logger.info(f"data: {cfg.train_tsv}")
    train_ds = DurationDataset(cfg.train_tsv, cfg.train_km, substring=cfg.substring)
    valid_ds = DurationDataset(cfg.valid_tsv, cfg.valid_km, substring=cfg.substring)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    best_loss = float("inf")
    for epoch in range(cfg.epochs):
        train_loss, train_loss_scaled = train_epoch(model, train_dl, l2_log_loss, optimizer, device)
        valid_loss, valid_loss_scaled, *acc = valid_epoch(model, valid_dl, l2_log_loss, device)
        acc0, acc1, acc2, acc3 = acc
        if valid_loss_scaled < best_loss:
            path = f"{os.getcwd()}/{cfg.substring}.ckpt"
            save_ckpt(model, path, cfg[cfg.model])
            best_loss = valid_loss_scaled
            logger.info(f"saved checkpoint: {path}")
            logger.info(f"[epoch {epoch}] train loss: {train_loss:.3f}, train scaled: {train_loss_scaled:.3f}")
            logger.info(f"[epoch {epoch}] valid loss: {valid_loss:.3f}, valid scaled: {valid_loss_scaled:.3f}")
            logger.info(f"acc: {acc0,acc1,acc2,acc3}")

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_loss_scaled = 0
    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        yhat = model(x)
        loss = criterion(yhat, y) * mask
        loss = torch.mean(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        yhat_scaled = torch.exp(yhat) - 1
        yhat_scaled = torch.round(yhat_scaled)
        scaled_loss = torch.mean(torch.abs(yhat_scaled - y) * mask)
        epoch_loss_scaled += scaled_loss.item()
    return epoch_loss / len(loader), epoch_loss_scaled / len(loader)

def valid_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_loss_scaled = 0
    acc = Accuracy()
    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        yhat = model(x)
        loss = criterion(yhat, y) * mask
        loss = torch.mean(loss)
        epoch_loss += loss.item()
        yhat_scaled = torch.exp(yhat) - 1
        yhat_scaled = torch.round(yhat_scaled)
        scaled_loss = torch.sum(torch.abs(yhat_scaled - y) * mask) / mask.sum()
        acc.update(yhat_scaled[mask].view(-1).float(), y[mask].view(-1).float())
        epoch_loss_scaled += scaled_loss.item()
    logger.info(f"example y: {y[0, :10].tolist()}")
    logger.info(f"example yhat: {yhat_scaled[0, :10].tolist()}")
    acc0 = acc.acc(tol=0)
    acc1 = acc.acc(tol=1)
    acc2 = acc.acc(tol=2)
    acc3 = acc.acc(tol=3)
    logger.info(f"accs: {acc0,acc1,acc2,acc3}")
    return epoch_loss / len(loader), epoch_loss_scaled / len(loader)

@hydra.main(config_path=".", config_name="duration_predictor.yaml")
def main(cfg):
    logger.info(f"{cfg}")
    train(cfg)

if __name__ == "__main__":
    main()