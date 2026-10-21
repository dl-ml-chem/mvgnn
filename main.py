import random
import numpy as np
import torch
import argparse

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from config import SEED, BATCH_SIZE, EPOCHS, TEST_SIZE, AUG_METHOD, AUG_FACTOR
from datasets.dataset import SolidElectrolyteDataset, build_train_test_datasets
from models.multiview_cgcnn import MultiViewCGCNN
from trainers.train_nn import train_model, evaluate_model


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Train MVGNN Model')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    seed_everything(SEED)

    # Use the helper to get properly scaled datasets
    full_dataset = SolidElectrolyteDataset(verbose=False)
    idx = list(range(len(full_dataset)))
    
    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    train_set, test_set = build_train_test_datasets(
        train_idx, 
        test_idx, 
        mode="multiview",
        aug_method=AUG_METHOD,
        aug_factor=AUG_FACTOR
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE
    )

    model = MultiViewCGCNN()

    # Pass test_loader as validation to monitor progress
    train_model(model, train_loader, EPOCHS, val_loader=test_loader, resume_path=args.resume)

    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()