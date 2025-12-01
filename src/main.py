import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader,Subset
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split

labels = np.load("../data/SoCs.npy") 
signal_data = np.load("../data/signal_data.npy")
# Path to spectrogram images
image_dir = "../data/waveform_images"
image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

# SoC valu es to remove
remove_soc_values = np.array([
    0.02702546, 0.7059901, 0.7331067,
    0.21593603, 0.12402861, 0.04449349
])
labels = np.round(labels, 6)
remove_soc_values = np.round(remove_soc_values, 6)
# Create mask for keeping samples (those NOT in remove list)
mask = ~np.isin(labels, remove_soc_values)

# Apply mask
labels = labels[mask]
signal_data = signal_data[mask]
image_filenames = [fname for i, fname in enumerate(image_filenames) if mask[i]]

# Sanity check
assert len(image_filenames) == len(labels), "Mismatch between number of images and labels."

# Image transformation (VGGNet expects normalized input)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class SpectrogramDataset(Dataset):
    def __init__(self, image_dir, image_filenames, labels, signals, transform=None):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.labels = labels
        self.signals = signals
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        signal = torch.tensor(self.signals[idx], dtype=torch.float32)
        return image, label, signal

# Create Dataset and DataLoader
dataset = SpectrogramDataset(image_dir, image_filenames, labels, signal_data, transform)

#TransferLearning Backbones
class VGG16Regressor(nn.Module):
    def __init__(self):
        super(VGG16Regressor, self).__init__()
        self.base = models.vgg16(pretrained=True)
        # for param in self.base.parameters():
        #     param.requires_grad = False
        self.base.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1)  # Regression output
        )
    
    def forward(self, x):
        return self.base(x)
class ResNet18Regressor(nn.Module):
    def __init__(self):
        super(ResNet18Regressor, self).__init__()
        self.base = models.resnet18(pretrained=True)

        # Replace the final classification layer (fc) for regression
        self.base.fc = nn.Sequential(
            nn.Linear(self.base.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Regression output
        )
    
    def forward(self, x):
        return self.base(x)
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.base = models.mobilenet_v2(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.base(x)
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.base = models.densenet121(pretrained=True)
        in_features = self.base.classifier.in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.base(x)
class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.base(x)

class UltrasoundDNN(nn.Module):
    def __init__(self, signal_length=30000):
        super(UltrasoundDNN, self).__init__()
        
        # Load the desired backbone for the waveform image branch
        self.base = models.resnet18(pretrained=True)
        img_feat_dim = self.base.fc.in_features

        # self.base = models.densenet121(pretrained=True)
        # img_feat_dim = self.base.classifier.in_features

        # self.base = models.mobilenet_v2(pretrained=True)
        # img_feat_dim = self.base.classifier[1].in_features

        # self.base = models.efficientnet_b0(pretrained=True)
        # img_feat_dim = self.base.classifier[1].in_features

        # self.base = models.vgg16(pretrained=True)
        # img_feat_dim = self.base.classifier[6].in_features

        # Replace classifier with identity (we will combine image & signal manually)
        # self.base.classifier[1] = nn.Identity()
        # use the above for the models except resnet

        self.base.fc = nn.Identity()

        # Signal branch
        self.signal_branch = nn.Sequential(
            nn.Linear(signal_length, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        # Combined regressor
        self.regressor = nn.Sequential(
            nn.Linear(img_feat_dim + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )


    def forward(self, image, signal):
        """
        image: shape [B, 3, H, W]
        signal: shape [B, 30000]
        """
        img_feat = self.base(image)
        signal_feat = self.signal_branch(signal)
        combined = torch.cat((img_feat, signal_feat), dim=1)
        return self.regressor(combined)


criterion = nn.MSELoss()

# ---------- User choice ----------
mode = input("Choose validation mode ('kfold' or 'holdout'): ").strip().lower()
if mode not in ('kfold', 'holdout'):
    raise ValueError("Invalid mode. Please type exactly 'kfold' or 'holdout'.")

# ---------- Config ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.MSELoss()

k_folds = 5
num_epochs = 100
batch_size = 12
random_state = 42

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

results_df = pd.DataFrame(columns=['Fold', 'Epoch', 'Train_Loss', 'MAE', 'RMSE', 'R2'])

predefined_colors = ['dodgerblue', 'orange', 'green', 'purple', 'red']

# Containers to collect best-epoch y_true/y_pred for plotting across folds
fold_y_true = []
fold_y_pred = []
fold_colors = []

# ---------- Utility: training + evaluation helper ----------
def train_and_evaluate(model, train_loader, test_loader, fold_idx, fold_color):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_mae = float('inf')
    best_y_true, best_y_pred = None, None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        print(f"🔁 Fold {fold_idx+1} | Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets, signals) in enumerate(train_loader):
            inputs = inputs.to(device)
            signals = signals.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs, signals)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = train_loss / len(train_loader)

        # Evaluation on test set
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, targets, signals in test_loader:
                inputs = inputs.to(device)
                signals = signals.to(device)
                targets = targets.to(device).unsqueeze(1)

                outputs = model(inputs, signals)
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        mae = mean_absolute_error(y_true, y_pred) if len(y_true) > 0 else np.nan
        rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) > 0 else np.nan
        r2 = r2_score(y_true, y_pred) if len(y_true) > 0 else np.nan

        print(f"📊 Fold {fold_idx+1}, Epoch {epoch+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        results_df.loc[len(results_df)] = [fold_idx+1, epoch+1, avg_loss, mae, rmse, r2]

        # Save best by MAE
        if not np.isnan(mae) and mae < best_mae:
            best_mae = mae
            ckpt_path = os.path.join(checkpoint_dir, f"fold_{fold_idx+1}_best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Fold {fold_idx+1} best model saved at epoch {epoch+1} with MAE={mae:.4f}")

            best_y_true = y_true.copy()
            best_y_pred = y_pred.copy()

    return best_y_true, best_y_pred

# ---------- Mode: K-Fold ----------
if mode == 'kfold':
    print(f"Running {k_folds}-fold cross-validation.")
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"\n📂 Fold {fold+1}/{k_folds}")
        fold_color = predefined_colors[fold % len(predefined_colors)]

        model = UltrasoundDNN().to(device)

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

        best_y_true, best_y_pred = train_and_evaluate(model, train_loader, test_loader, fold, fold_color)

        # store if available
        if best_y_true is not None and best_y_pred is not None and len(best_y_true) > 0:
            fold_y_true.append(best_y_true)
            fold_y_pred.append(best_y_pred)
            fold_colors.append(np.array([fold_color] * len(best_y_true)))

# ---------- Mode: Holdout 80/20 ----------
else:
    print("Running single 80/20 hold-out split (train/val).")
    # If your dataset is indexable and dataset[i] returns (input, target, signal)
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=random_state, shuffle=True)

    model = UltrasoundDNN().to(device)

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    fold_color = predefined_colors[0]
    best_y_true, best_y_pred = train_and_evaluate(model, train_loader, test_loader, 0, fold_color)

    if best_y_true is not None and best_y_pred is not None and len(best_y_true) > 0:
        fold_y_true.append(best_y_true)
        fold_y_pred.append(best_y_pred)
        fold_colors.append(np.array([fold_color] * len(best_y_true)))

# ---------- After training: prepare combined results ----------
if len(fold_y_true) == 0:
    raise RuntimeError("No predictions collected from any fold. Check training/evaluation loops.")

all_y_true = np.concatenate(fold_y_true)
all_y_pred = np.concatenate(fold_y_pred)
all_colors = np.concatenate(fold_colors)

# If you have a boolean mask to filter samples, apply it only if mask exists and length matches
if 'mask' in globals() and isinstance(mask, (np.ndarray, list)) and len(mask) == len(all_y_true):
    mask_arr = np.array(mask, dtype=bool)
    all_y_true = all_y_true[mask_arr]
    all_y_pred = all_y_pred[mask_arr]
    all_colors = all_colors[mask_arr]
else:
    # if mask undefined or mismatched, just skip filtering
    pass

# Plot: Actual vs Predicted
plt.figure(figsize=(10, 10))
plt.scatter(all_y_true, all_y_pred, color=all_colors, alpha=0.7, label='Predictions')
plt.plot([all_y_true.min(), all_y_true.max()],
         [all_y_true.min(), all_y_true.max()],
         'r--', lw=2, label='Perfect Fit (y = x)')
plt.xlabel('Actual SoC', fontsize=16)
plt.ylabel('Predicted SoC', fontsize=16)
plt.title('Prediction vs Actual Values (All Folds)', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('prediction_vs_actual.png', dpi=300)
print("\n📈 Prediction vs Actual Plot saved as 'prediction_vs_actual.png'.")

# Save results table
results_df.to_excel("results.xlsx", index=False)
print("📊 All results saved to 'results.xlsx'.")