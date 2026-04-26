"""
train_model.py — Fine-tune EfficientNet B0 on Askari receipt dataset
Run once before launching the main app.

Usage:
    python train_model.py --data path/to/data

Expected folder structure:
    data/
      real/    ← authentic Askari receipts
      Fake/    ← fake / AI-generated receipts

Output:
    askari_efficientnet.pth  (saved in the same directory as this script)
"""

import os
import sys
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_SAVE_PATH = "askari_efficientnet.pth"
EPOCHS          = 80
LR              = 1e-4
BATCH_SIZE      = 4
SEED            = 42

torch.manual_seed(SEED)
random.seed(SEED)


# ─────────────────────────────────────────────
#  AUGMENTATION
#  Heavy augmentation is critical with only 8 images.
#  We simulate: varying phone angles, lighting, zoom,
#  WhatsApp compression, screen glare, and partial crops.
# ─────────────────────────────────────────────
TRAIN_TRANSFORM = T.Compose([
    T.Resize((320, 320)),
    T.RandomCrop(224),                                   # simulate different zoom / framing
    T.RandomRotation(degrees=8),                         # slight tilt (hand-held photo)
    T.RandomHorizontalFlip(p=0.2),                       # occasional mirror
    T.ColorJitter(brightness=0.4, contrast=0.4,          # screen brightness / ambient light
                  saturation=0.3, hue=0.05),
    T.RandomGrayscale(p=0.05),                           # rare grayscale screenshot
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),     # camera soft-focus
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.25, scale=(0.01, 0.08),          # simulate partial obstruction
                    ratio=(0.3, 3.0)),
])

EVAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
#  BUILD MODEL
#  Replace EfficientNet's 1000-class head with a
#  2-class (fake / real) binary classifier.
# ─────────────────────────────────────────────
def build_model():
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = efficientnet_b0(weights=weights)

    # Freeze all backbone layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 feature blocks for domain adaptation
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    # Replace classifier: 1280 → 256 → 2
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 2),
    )
    return model


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train(data_root: str):
    # ImageFolder sorts class names alphabetically:
    # 'Fake' < 'real'  →  class_to_idx = {'Fake': 0, 'real': 1}
    # We confirm this below and save it so inference knows the mapping.
    dataset = ImageFolder(root=data_root, transform=TRAIN_TRANSFORM)
    print(f"Classes found: {dataset.class_to_idx}")
    print(f"Total samples (before augmentation weighting): {len(dataset)}")

    # Weighted sampler to balance the 2:6 class imbalance
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        class_counts[label] += 1

    class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    sampler = WeightedRandomSampler(sample_weights,
                                     num_samples=len(dataset) * 20,  # oversample
                                     replacement=True)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

    model     = build_model()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # Up-weight the minority class (real) in the loss
    real_idx    = dataset.class_to_idx.get('real', 1)
    fake_idx    = dataset.class_to_idx.get('Fake', 0)
    weight_vec  = torch.zeros(2)
    weight_vec[real_idx] = 3.0   # real receipts are rare → penalise missing them
    weight_vec[fake_idx] = 1.0
    criterion   = nn.CrossEntropyLoss(weight=weight_vec)

    model.train()
    print(f"\nTraining for {EPOCHS} epochs …\n")
    for epoch in range(1, EPOCHS + 1):
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds   = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        acc = correct / total * 100
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={total_loss/len(loader):.4f} | acc={acc:.1f}%")

    # Final evaluation on the full dataset (no augmentation)
    eval_dataset = ImageFolder(root=data_root, transform=EVAL_TRANSFORM)
    eval_loader  = DataLoader(eval_dataset, batch_size=8, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in eval_loader:
            preds   = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    final_acc = correct / total * 100
    print(f"\nFinal training accuracy: {final_acc:.1f}%  ({correct}/{total})")

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx'    : dataset.class_to_idx,
        'final_accuracy'  : final_acc,
        'epochs'          : EPOCHS,
    }, MODEL_SAVE_PATH)
    print(f"\nModel saved → {MODEL_SAVE_PATH}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data",
                        help="Path to folder containing 'real' and 'Fake' subfolders")
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(f"ERROR: '{args.data}' is not a valid directory.")
        sys.exit(1)

    train(args.data)
