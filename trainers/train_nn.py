import os
import math
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import DEVICE, LR, CHECKPOINT_DIR


class EarlyStopping:
    def __init__(self, patience=50, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            self.counter = 0

    def load_state_dict(self, state_dict):
        self.patience = state_dict['patience']
        self.min_delta = state_dict['min_delta']
        self.counter = state_dict['counter']
        self.best_loss = state_dict['best_loss']
        self.early_stop = state_dict['early_stop']
        self.best_state = state_dict['best_state']

    def state_dict(self):
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'counter': self.counter,
            'best_loss': self.best_loss,
            'early_stop': self.early_stop,
            'best_state': self.best_state
        }


def train_model(model, train_loader, epochs, val_loader=None, resume_path=None):

    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    early_stopping = EarlyStopping(patience=60, verbose=True)
    loss_fn = nn.MSELoss()

    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        early_stopping.load_state_dict(checkpoint['early_stopping_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}")

    print(f"Starting training on {DEVICE}...")

    for epoch in range(start_epoch, epochs):

        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            _, pred = model(batch)
            loss = loss_fn(pred, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        monitor_loss = avg_train_loss
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for v_batch in val_loader:
                    v_batch = v_batch.to(DEVICE)
                    _, v_pred = model(v_batch)
                    v_loss = loss_fn(v_pred, v_batch.y)
                    val_loss += v_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            monitor_loss = avg_val_loss
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(monitor_loss)
        
        # Save per-epoch temporary model
        checkpoint_name = f"checkpoint_epoch{epoch}_loss{avg_train_loss:.4f}.pt"
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'early_stopping_state_dict': early_stopping.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss if val_loader else None
        }
        torch.save(checkpoint_data, checkpoint_path)

        # Check early stopping and save best model
        old_best_loss = early_stopping.best_loss
        early_stopping(monitor_loss, model)
        
        if early_stopping.best_loss != old_best_loss:
            best_model_path = os.path.join(CHECKPOINT_DIR, "model_best.pt")
            torch.save(checkpoint_data, best_model_path)
            print(f"New best model saved at epoch {epoch} with monitor loss {monitor_loss:.4f}")

        if early_stopping.early_stop:
            print("Early stopping triggered. Training halted.")
            break

    # Load best model weights
    if early_stopping.best_state:
        model.load_state_dict(early_stopping.best_state)
        print("Restored best model weights.")


def evaluate_model(model, loader):

    model.eval()

    ys = []
    ps = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(DEVICE)

            _, pred = model(batch)

            ys.extend(batch.y.cpu().numpy().flatten())
            ps.extend(pred.cpu().numpy().flatten())

    mae = mean_absolute_error(ys, ps)
    rmse = math.sqrt(mean_squared_error(ys, ps))
    r2 = r2_score(ys, ps)

    print("\nTest Results")
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))

    return mae, rmse, r2