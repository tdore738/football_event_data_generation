import os
import json
import copy
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# 0. USER CONFIG
# =========================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-3
WEIGHT_DECAY = 0.0

ROOT_SAVE_DIR = Path("cnn_experiments")
ROOT_SAVE_DIR.mkdir(exist_ok=True, parents=True)

# Example: metadata CSV with columns:
# image_path,label,Match,Frame
METADATA_CSV = "frames_metadata.csv"

# Set this to your target image size
IMG_H = 90
IMG_W = 160


# =========================================================
# 1. REPRODUCIBILITY
# =========================================================

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================================================
# 2. IMAGE REPRESENTATIONS
# =========================================================

def ensure_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def rep_full_color(img_bgr):
    # output: H x W x 3
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rep_full_color_blurred(img_bgr):
    img = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rep_gray(img_bgr):
    # output: H x W
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def rep_gray_blurred(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def rep_edge_map(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def rep_blur_then_edge(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    return edges


REPRESENTATIONS = {
    "full_color": {
        "fn": rep_full_color,
        "in_channels": 3,
    },
    "full_color_blurred": {
        "fn": rep_full_color_blurred,
        "in_channels": 3,
    },
    "gray": {
        "fn": rep_gray,
        "in_channels": 1,
    },
    "gray_blurred": {
        "fn": rep_gray_blurred,
        "in_channels": 1,
    },
    "edge_map": {
        "fn": rep_edge_map,
        "in_channels": 1,
    },
    "blur_then_edge": {
        "fn": rep_blur_then_edge,
        "in_channels": 1,
    },
}


# =========================================================
# 3. DATASET
# =========================================================

class FrameDataset(Dataset):
    def __init__(self, df, representation_fn, img_h=90, img_w=160):
        self.df = df.reset_index(drop=True).copy()
        self.representation_fn = representation_fn
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        label = int(row["label"])

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # resize before representation, or after — choose one and keep it consistent
        img_bgr = cv2.resize(img_bgr, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)

        img = self.representation_fn(img_bgr)
        img = ensure_uint8(img)

        # convert to float tensor in [0,1]
        if img.ndim == 2:
            # grayscale-like: H x W -> 1 x H x W
            x = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        elif img.ndim == 3:
            # RGB: H x W x C -> C x H x W
            x = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        y = torch.tensor(label, dtype=torch.float32)

        meta = {
            "Match": row["Match"],
            "Frame": int(row["Frame"]),
            "image_path": img_path,
        }

        return x, y, meta


# =========================================================
# 4. MODEL DEFINITIONS
#    Paste your 5 models here.
#    You need 1-channel and 3-channel versions, or make input channels configurable.
# =========================================================

class NetTiny(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ---- Replace / paste the other 4 models from earlier here ----
# Example stubs:
class NetBaseline(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class NetRegularized(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class NetGentleDownsample(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class NetLargeKernel(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


MODEL_FACTORIES = {
    "NetTiny": NetTiny,
    "NetBaseline": NetBaseline,
    "NetRegularized": NetRegularized,
    "NetGentleDownsample": NetGentleDownsample,
    "NetLargeKernel": NetLargeKernel,
}


# =========================================================
# 5. TRAIN / EVAL HELPERS
# =========================================================

def make_loaders(train_df, val_df, rep_name):
    rep_fn = REPRESENTATIONS[rep_name]["fn"]

    train_ds = FrameDataset(train_df, rep_fn, img_h=IMG_H, img_w=IMG_W)
    val_ds = FrameDataset(val_df, rep_fn, img_h=IMG_H, img_w=IMG_W)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        logits = model(inputs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float()

        logits = model(inputs).squeeze(1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        running_loss += loss.item()
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())
        all_preds.extend(preds.cpu().numpy().astype(int).tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
    }


def save_checkpoint(path, model, optimizer, epoch, train_metrics, val_metrics):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    torch.save(ckpt, path)


# =========================================================
# 6. MAIN TRAINING ROUTINE FOR ONE (holdout, representation, model)
# =========================================================

def run_one_experiment(train_df, val_df, rep_name, model_name, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    in_channels = REPRESENTATIONS[rep_name]["in_channels"]
    model = MODEL_FACTORIES[model_name](in_channels=in_channels).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader, val_loader = make_loaders(train_df, val_df, rep_name)

    history = []
    best_val_loss = float("inf")
    best_epoch = None
    no_improve_count = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)

        print(
            f"[{model_name} | {rep_name}] "
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        # save every epoch
        epoch_ckpt = save_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            epoch_ckpt,
            model,
            optimizer,
            epoch,
            train_metrics={"loss": train_loss},
            val_metrics={"loss": val_metrics["loss"], "accuracy": val_metrics["accuracy"]},
        )

        # best model tracking
        if val_metrics["loss"] < best_val_loss - MIN_DELTA:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            no_improve_count = 0

            best_ckpt = save_dir / "best.pt"
            save_checkpoint(
                best_ckpt,
                model,
                optimizer,
                epoch,
                train_metrics={"loss": train_loss},
                val_metrics={"loss": val_metrics["loss"], "accuracy": val_metrics["accuracy"]},
            )
        else:
            no_improve_count += 1

        # save history after each epoch too
        pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)

        if no_improve_count >= PATIENCE:
            print(
                f"Early stopping triggered for {model_name} | {rep_name} "
                f"at epoch {epoch}, best epoch was {best_epoch}"
            )
            break

    # final summary
    summary = {
        "model_name": model_name,
        "representation": rep_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history),
        "n_train": len(train_df),
        "n_val": len(val_df),
    }

    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# =========================================================
# 7. HOLDOUT LOOP
# =========================================================

def main():
    df = pd.read_csv(METADATA_CSV)

    required_cols = {"image_path", "label", "Match", "Frame"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")

    all_matches = sorted(df["Match"].unique().tolist())

    # all possible 2-match holdouts
    holdout_pairs = list(combinations(all_matches, 2))

    all_results = []

    for holdout_pair in holdout_pairs:
        holdout_pair = tuple(sorted(holdout_pair))
        holdout_name = "__".join(holdout_pair)

        print("=" * 80)
        print(f"HOLDOUT: {holdout_pair}")
        print("=" * 80)

        train_df = df[~df["Match"].isin(holdout_pair)].copy()
        val_df = df[df["Match"].isin(holdout_pair)].copy()

        # sort for sanity / reproducibility
        train_df = train_df.sort_values(["Match", "Frame"]).reset_index(drop=True)
        val_df = val_df.sort_values(["Match", "Frame"]).reset_index(drop=True)

        for rep_name in REPRESENTATIONS.keys():
            for model_name in MODEL_FACTORIES.keys():
                save_dir = (
                    ROOT_SAVE_DIR
                    / f"holdout_{holdout_name}"
                    / rep_name
                    / model_name
                )

                try:
                    summary = run_one_experiment(
                        train_df=train_df,
                        val_df=val_df,
                        rep_name=rep_name,
                        model_name=model_name,
                        save_dir=save_dir,
                    )
                    summary["holdout_pair"] = list(holdout_pair)
                    all_results.append(summary)

                except Exception as e:
                    print(f"FAILED: holdout={holdout_pair}, rep={rep_name}, model={model_name}")
                    print(repr(e))

                    fail_record = {
                        "holdout_pair": list(holdout_pair),
                        "representation": rep_name,
                        "model_name": model_name,
                        "status": "failed",
                        "error": repr(e),
                    }
                    all_results.append(fail_record)

        # save partial results after each holdout
        pd.DataFrame(all_results).to_csv(ROOT_SAVE_DIR / "all_results_so_far.csv", index=False)

    # final table
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(ROOT_SAVE_DIR / "all_results_final.csv", index=False)

    print("Done.")
    print(results_df.head())


if __name__ == "__main__":
    main()