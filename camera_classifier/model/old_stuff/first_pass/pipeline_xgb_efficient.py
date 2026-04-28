import gc
import json
import warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# CONFIG
# =========================================================

FRAMES_METADATA_CSV = "frames_metadata.csv"

CNN_RUNS_ROOT = Path(
    "/home/travis/Projects/football_event_data_generation/camera_classifier/model/first_pass/cnn_experiments"
)

XGB_OUTPUT_ROOT = Path(
    "/home/travis/Projects/football_event_data_generation/camera_classifier/model/first_pass/xgb_experiments"
)
XGB_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
IMG_H = 90
IMG_W = 160
EPOCH_STRIDE = 1

EVALUATE_ALL_RUNS = True
SINGLE_RUN_DIR = None

RAW_XGB_MODEL_NAME = "RawXGB"

# Since data is preloaded in RAM, workers usually do not help much.
NUM_WORKERS_IN_MEMORY = 0
PIN_MEMORY = torch.cuda.is_available()

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch.load` with `weights_only=False`",
)


# =========================================================
# REPRESENTATIONS
# =========================================================

def ensure_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def rep_full_color(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rep_full_color_blurred(img_bgr):
    img = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rep_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def rep_gray_blurred(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def rep_edge_map(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150)


def rep_blur_then_edge(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 50, 150)


REPRESENTATIONS = {
    "full_color": {"fn": rep_full_color, "in_channels": 3},
    "full_color_blurred": {"fn": rep_full_color_blurred, "in_channels": 3},
    "gray": {"fn": rep_gray, "in_channels": 1},
    "gray_blurred": {"fn": rep_gray_blurred, "in_channels": 1},
    "edge_map": {"fn": rep_edge_map, "in_channels": 1},
    "blur_then_edge": {"fn": rep_blur_then_edge, "in_channels": 1},
}


# =========================================================
# MODEL DEFINITIONS
# compatible with your training script
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
# PRELOADED UINT8 DATASET
# X_uint8 shape:
#   grayscale-like -> [N, H, W]
#   color          -> [N, H, W, 3]
# =========================================================

class PreloadedUint8Dataset(Dataset):
    def __init__(self, X_uint8, y_array, indices):
        self.X_uint8 = X_uint8
        self.y_array = y_array
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        j = self.indices[idx]
        x = self.X_uint8[j]

        if x.ndim == 2:
            x_t = torch.from_numpy(x).float().unsqueeze(0) / 255.0
        elif x.ndim == 3:
            x_t = torch.from_numpy(x).float().permute(2, 0, 1) / 255.0
        else:
            raise ValueError(f"Unexpected cached shape: {x.shape}")

        y = self.y_array[j]
        return x_t, y


def make_preloaded_loader(X_uint8, y_array, indices):
    ds = PreloadedUint8Dataset(X_uint8, y_array, indices)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_IN_MEMORY,
        pin_memory=PIN_MEMORY,
    )


# =========================================================
# FEATURE EXTRACTION
# =========================================================

@torch.no_grad()
def encode_features(model, x):
    x = model.features(x)
    x = model.pool(x)

    if isinstance(model, NetRegularized):
        x = model.classifier[:-2](x)
    else:
        x = model.classifier[:-1](x)

    return x


@torch.no_grad()
def extract_cnn_features_from_preloaded(model, loader):
    model.eval()
    X_chunks = []
    y_chunks = []

    for inputs, labels in loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        z = encode_features(model, inputs).cpu().numpy()
        X_chunks.append(z)
        y_chunks.append(np.asarray(labels))

    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    return X, y


def flatten_raw_subset(X_uint8, y_array, indices):
    idx = np.asarray(indices, dtype=np.int64)
    X = X_uint8[idx].reshape(len(idx), -1)
    y = y_array[idx]
    return X, y


# =========================================================
# BUILD REPRESENTATION CACHE ONCE PER REPRESENTATION
# Stores uint8, not float32
# =========================================================

def build_representation_uint8(frames_df, representation):
    rep_fn = REPRESENTATIONS[representation]["fn"]

    rows = []
    for i, row in frames_df.iterrows():
        img_path = row["image_path"]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img_bgr = cv2.resize(img_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        img = ensure_uint8(rep_fn(img_bgr))
        rows.append(img)

        if (i + 1) % 1000 == 0:
            print(f"    preloaded {i+1}/{len(frames_df)} images", end="\r")

    X_uint8 = np.stack(rows, axis=0)
    y_array = frames_df["label"].to_numpy().astype(np.int64)

    print(f"    preloaded {len(frames_df)}/{len(frames_df)} images")
    print(f"    cached dtype={X_uint8.dtype}, shape={X_uint8.shape}")
    return X_uint8, y_array


# =========================================================
# RUN DISCOVERY / PARSING
# =========================================================

def find_run_dirs(root: Path):
    run_dirs = []
    for p in root.rglob("*"):
        if p.is_dir() and any(p.glob("epoch_*.pt")):
            run_dirs.append(p)
    return sorted(run_dirs)


def parse_run_dir(run_dir: Path):
    model_name = run_dir.name
    representation = run_dir.parent.name
    holdout_folder = run_dir.parent.parent.name

    if not holdout_folder.startswith("holdout_"):
        raise ValueError(f"Unexpected holdout folder: {holdout_folder}")

    holdout_matches = holdout_folder[len("holdout_"):].split("__")
    return holdout_matches, representation, model_name


def output_dir_for_run(run_dir: Path) -> Path:
    rel = run_dir.relative_to(CNN_RUNS_ROOT)
    out_dir = XGB_OUTPUT_ROOT / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def output_dir_for_raw_xgb(holdout_matches, representation) -> Path:
    holdout_folder = "holdout_" + "__".join(holdout_matches)
    out_dir = XGB_OUTPUT_ROOT / holdout_folder / representation / RAW_XGB_MODEL_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# =========================================================
# XGBOOST
# =========================================================

def make_xgb():
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        nthread=-1,
    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate_raw_xgb_from_preloaded(X_uint8, y_array, train_idx, test_idx, representation, holdout_matches):
    X_train, y_train = flatten_raw_subset(X_uint8, y_array, train_idx)
    X_test, y_test = flatten_raw_subset(X_uint8, y_array, test_idx)

    xgb = make_xgb()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    result = {
        "epoch": -1,
        "epoch_file": "raw_input",
        "model_name": RAW_XGB_MODEL_NAME,
        "representation": representation,
        "holdout_matches": "|".join(holdout_matches),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "feature_dim": int(X_train.shape[1]),
        "xgb_accuracy": float(acc),
    }

    del X_train, X_test, y_train, y_test, y_pred, xgb
    gc.collect()
    return result


def evaluate_checkpoint_from_preloaded(run_dir, ckpt_path, X_uint8, y_array, train_idx, test_idx):
    holdout_matches, representation, model_name = parse_run_dir(run_dir)
    in_channels = REPRESENTATIONS[representation]["in_channels"]

    model = MODEL_FACTORIES[model_name](in_channels=in_channels).to(DEVICE)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch_num = checkpoint.get("epoch", -1)

    train_loader = make_preloaded_loader(X_uint8, y_array, train_idx)
    test_loader = make_preloaded_loader(X_uint8, y_array, test_idx)

    X_train, y_train = extract_cnn_features_from_preloaded(model, train_loader)
    X_test, y_test = extract_cnn_features_from_preloaded(model, test_loader)

    xgb = make_xgb()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    result = {
        "epoch": epoch_num,
        "epoch_file": ckpt_path.name,
        "model_name": model_name,
        "representation": representation,
        "holdout_matches": "|".join(holdout_matches),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "feature_dim": int(X_train.shape[1]),
        "xgb_accuracy": float(acc),
    }

    del X_train, X_test, y_train, y_test, y_pred, xgb, model, checkpoint, train_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# =========================================================
# MAIN
# =========================================================

def main():
    frames_df = pd.read_csv(FRAMES_METADATA_CSV)

    required_cols = {"image_path", "label", "Match", "Frame"}
    missing = required_cols - set(frames_df.columns)
    if missing:
        raise ValueError(f"frames metadata missing columns: {missing}")

    if EVALUATE_ALL_RUNS:
        run_dirs = find_run_dirs(CNN_RUNS_ROOT)
    else:
        run_dirs = [SINGLE_RUN_DIR]

    print(f"\nFound {len(run_dirs)} CNN run directories.")
    print(f"Source : {CNN_RUNS_ROOT}")
    print(f"Output : {XGB_OUTPUT_ROOT}\n")

    rep_to_runs = defaultdict(list)
    for run_dir in run_dirs:
        holdout_matches, representation, model_name = parse_run_dir(run_dir)
        rep_to_runs[representation].append(run_dir)

    all_results = []

    for rep_idx, representation in enumerate(sorted(rep_to_runs.keys()), start=1):
        print("=" * 96)
        print(f"[REP {rep_idx}/{len(rep_to_runs)}] {representation}")
        print("Preloading representation once as uint8...")
        print("=" * 96)

        X_uint8, y_array = build_representation_uint8(frames_df, representation)
        runs_this_rep = sorted(rep_to_runs[representation])

        unique_holdouts = sorted({
            tuple(parse_run_dir(run_dir)[0]) for run_dir in runs_this_rep
        })

        print(f"\nRunning {len(unique_holdouts)} raw-XGBoost baselines for {representation}...\n")

        for raw_idx, holdout_matches in enumerate(unique_holdouts, start=1):
            holdout_matches = list(holdout_matches)

            match_mask = frames_df["Match"].isin(holdout_matches).to_numpy()
            train_idx = np.where(~match_mask)[0]
            test_idx = np.where(match_mask)[0]

            out_dir = output_dir_for_raw_xgb(holdout_matches, representation)

            print("-" * 88)
            print(f"[RAW {raw_idx}/{len(unique_holdouts)}] {representation}")
            print(f"holdout : {', '.join(holdout_matches)}")
            print(f"save    : {out_dir}")
            print("-" * 88)

            try:
                result = evaluate_raw_xgb_from_preloaded(
                    X_uint8, y_array, train_idx, test_idx, representation, holdout_matches
                )
                all_results.append(result)

                pd.DataFrame([result]).to_csv(out_dir / "xgb_results_by_epoch.csv", index=False)
                with open(out_dir / "xgb_summary.json", "w") as f:
                    json.dump(result, f, indent=2)

                print(f"  raw baseline acc = {result['xgb_accuracy']:.4f}\n")

            except Exception as e:
                print(f"  FAILED: {e}\n")

        print(f"\nRunning {len(runs_this_rep)} CNN-feature sweeps for {representation}...\n")

        for run_idx, run_dir in enumerate(runs_this_rep, start=1):
            holdout_matches, _, model_name = parse_run_dir(run_dir)
            out_dir = output_dir_for_run(run_dir)

            match_mask = frames_df["Match"].isin(holdout_matches).to_numpy()
            train_idx = np.where(~match_mask)[0]
            test_idx = np.where(match_mask)[0]

            ckpts = sorted(run_dir.glob("epoch_*.pt"))[::EPOCH_STRIDE]
            if not ckpts:
                continue

            print("-" * 88)
            print(f"[{run_idx}/{len(runs_this_rep)}] {model_name} | {representation}")
            print(f"holdout : {', '.join(holdout_matches)}")
            print(f"epochs  : {len(ckpts)}")
            print(f"save    : {out_dir}")
            print("-" * 88)

            run_results = []

            for i, ckpt_path in enumerate(ckpts, start=1):
                try:
                    result = evaluate_checkpoint_from_preloaded(
                        run_dir, ckpt_path, X_uint8, y_array, train_idx, test_idx
                    )
                    run_results.append(result)
                    all_results.append(result)

                    print(
                        f"  [{i:>3}/{len(ckpts)}] "
                        f"{ckpt_path.name:12s} "
                        f"acc={result['xgb_accuracy']:.4f}"
                    )

                except Exception as e:
                    print(
                        f"  [{i:>3}/{len(ckpts)}] "
                        f"{ckpt_path.name:12s} "
                        f"FAILED: {e}"
                    )

            if run_results:
                run_df = pd.DataFrame(run_results).sort_values("epoch")
                run_df.to_csv(out_dir / "xgb_results_by_epoch.csv", index=False)

                best_row = run_df.sort_values("xgb_accuracy", ascending=False).iloc[0].to_dict()
                with open(out_dir / "xgb_summary.json", "w") as f:
                    json.dump(best_row, f, indent=2)

                print(
                    f"  best epoch: {int(best_row['epoch'])} | "
                    f"best acc: {best_row['xgb_accuracy']:.4f}\n"
                )

        # free this representation before next one
        del X_uint8, y_array
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if all_results:
        all_df = pd.DataFrame(all_results)
        all_df.to_csv(XGB_OUTPUT_ROOT / "all_xgb_results.csv", index=False)

        top_df = all_df.sort_values("xgb_accuracy", ascending=False).head(25)
        top_df.to_csv(XGB_OUTPUT_ROOT / "top_25_xgb_results.csv", index=False)

        print("\nTop results:\n")
        print(
            top_df[
                ["model_name", "representation", "holdout_matches", "epoch", "xgb_accuracy"]
            ].to_string(index=False)
        )
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()