# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os
import time
import numpy as np
from pathlib import Path
import traceback

import matplotlib.pyplot as plt
# *** ADD SEABORN FOR NICER PLOTS (OPTIONAL BUT RECOMMENDED) ***
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid") # Set a nice theme
    USE_SEABORN = True
except ImportError:
    USE_SEABORN = False
    print("[!] Seaborn not found. Plots will use default matplotlib style.")
# ***********************************************************

# --- Configuration ---
DATASET_BASE_DIR = Path("./")
# (Hyperparameters remain the same)
BATCH_SIZE = 32; NUM_EPOCHS = 30; LEARNING_RATE = 1e-4; WEIGHT_DECAY = 1e-4
STEP_LR_STEP_SIZE = 7; STEP_LR_GAMMA = 0.1; NUM_CLASSES = 200
MODEL_SAVE_NAME = "cub_bird_classifier_resnet50_finetuned.pth"
INPUT_SIZE = 224; USE_BOUNDING_BOXES = True; FREEZE_PRETRAINED_WEIGHTS = False

# --- Plotting Configuration ---
PLOT_SAVE_DIR = Path("./training_plots") # Directory to save plots
PLOT_LOSS_FILENAME = "loss_history.png"
PLOT_ACC_FILENAME = "accuracy_history.png"
# *** NEW PLOT FILENAMES ***
PLOT_LR_FILENAME = "learning_rate_schedule.png"
PLOT_CLASSDIST_FILENAME = "train_class_distribution.png"
PLOT_BBOXAREA_FILENAME = "bounding_box_area_distribution.png"
# **************************


# --- Setup ---
# (Dataset path setup remains the same)
if (DATASET_BASE_DIR / "CUB_200_2011").is_dir():
    DATASET_PATH = DATASET_BASE_DIR / "CUB_200_2011"
elif DATASET_BASE_DIR.name == "CUB_200_2011" and DATASET_BASE_DIR.is_dir():
     DATASET_PATH = DATASET_BASE_DIR
else:
    raise FileNotFoundError(f"CUB_200_2011 dataset not found in {DATASET_BASE_DIR}")

IMAGES_DIR = DATASET_PATH / "images"
print(f"[*] Using dataset path: {DATASET_PATH}")
print(f"[*] Images directory: {IMAGES_DIR}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}")

# --- Data Loading and Preprocessing ---
# (load_metadata function remains largely the same, ensure it returns classes_df)
def load_metadata(dataset_path):
    # ... (previous implementation using r'\s+') ...
    # Make sure classes_df is correctly loaded and returned
    print("[*] Loading metadata files...")
    essential_files_ok = True
    try:
        images_df = pd.read_csv(dataset_path/'images.txt', sep=' ', names=['img_id', 'filepath'], header=None)
        labels_df = pd.read_csv(dataset_path/'image_class_labels.txt', sep=' ', names=['img_id', 'class_id'], header=None)
        split_df = pd.read_csv(dataset_path/'train_test_split.txt', sep=' ', names=['img_id', 'is_train'], header=None)
        bounding_boxes_df = pd.read_csv(dataset_path/'bounding_boxes.txt', sep=' ', names=['img_id', 'x', 'y', 'width', 'height'], header=None)
        # Load classes_df, needed for class distribution plot labels later
        classes_df = pd.read_csv(dataset_path/'classes.txt', sep=r'\s+', names=['class_id', 'class_name'], header=None)
        print(f"    - Loaded {len(classes_df)} class names.")
    except FileNotFoundError as e:
        print(f"[!] Error loading essential metadata file: {e}")
        essential_files_ok = False
        images_df, labels_df, split_df, bounding_boxes_df, classes_df = [pd.DataFrame()]*5

    # ... (loading optional parts/attributes remains the same) ...
    part_locs_df=None; image_attributes_df=None # Simplified for brevity
    try:
        if (p:=dataset_path / 'parts/part_locs.txt').exists(): part_locs_df = pd.read_csv(p, sep=r'\s+', names=['img_id','part_id','x','y','visible'], header=None, on_bad_lines='warn'); print("    - Part locations loaded.")
        else: print("    - Info: parts/part_locs.txt not found.")
    except Exception as e: print(f"[!] Warning: Could not read part_locs.txt: {e}")
    try:
        if (p:=dataset_path / 'attributes/image_attribute_labels.txt').exists(): image_attributes_df = pd.read_csv(p, sep=r'\s+', names=['img_id','attribute_id','is_present','certainty_id','time'], header=None, on_bad_lines='warn'); print("    - Image attributes loaded.")
        else: print("    - Info: attributes/image_attribute_labels.txt not found.")
    except Exception as e: print(f"[!] Warning: Could not read image_attribute_labels.txt: {e}")


    if not essential_files_ok: raise RuntimeError("Essential metadata files could not be loaded.")

    print("[*] Merging metadata...")
    data_df = pd.merge(images_df, labels_df, on='img_id')
    data_df = pd.merge(data_df, split_df, on='img_id')
    data_df = pd.merge(data_df, bounding_boxes_df, on='img_id')
    data_df['full_path'] = data_df['filepath'].apply(lambda x: IMAGES_DIR / x)
    data_df['class_id'] = data_df['class_id'] - 1 # 0-indexed
    print(f"    - Adjusted class IDs to be 0-indexed (0 to {data_df['class_id'].max()}).")
    print("[*] Metadata loaded and merged successfully.")
    return data_df, classes_df # *** RETURN classes_df ***


# --- CUBDataset Class (remains the same) ---
class CUBDataset(Dataset): # (Implementation as before)
    def __init__(self, data_df, transform=None, use_bounding_box=True):
        self.data_df = data_df; self.transform = transform; self.use_bounding_box = use_bounding_box
        print(f"[*] Dataset initialized: {len(self.data_df)} samples. BBox: {self.use_bounding_box}")
    def __len__(self): return len(self.data_df)
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        img_info = self.data_df.iloc[idx]; img_path = img_info['full_path']; label = img_info['class_id']
        try:
            image = Image.open(img_path).convert('RGB')
            if self.use_bounding_box:
                bbox = img_info[['x', 'y', 'width', 'height']].values.astype(float)
                W, H = image.size; l=max(0, int(np.floor(bbox[0]))); u=max(0, int(np.floor(bbox[1])))
                r=min(W, int(np.ceil(bbox[0]+bbox[2]))); b=min(H, int(np.ceil(bbox[1]+bbox[3])))
                if r > l and b > u: image = image.crop((l, u, r, b))
            if self.transform: image = self.transform(image)
            return image, label
        except (FileNotFoundError, UnidentifiedImageError) as e: print(f"[!] Img Err {img_path}: {e}. Skip."); return self.__getitem__((idx + 1) % len(self))
        except Exception as e: print(f"[!] Proc Err {img_path}: {e}"); traceback.print_exc(); return self.__getitem__((idx + 1) % len(self))


# --- Define Transformations (remains the same) ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([ # (As before)
    transforms.Resize((INPUT_SIZE+32, INPUT_SIZE+32)), transforms.RandomRotation(20),
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(), normalize,
])
test_transform = transforms.Compose([ # (As before)
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(), normalize,
])

# --- Load Data and Create Datasets/DataLoaders ---
# *** CAPTURE classes_df HERE ***
all_data_df, classes_df = load_metadata(DATASET_PATH)
# *****************************
train_df = all_data_df[all_data_df['is_train'] == 1].reset_index(drop=True)
test_df = all_data_df[all_data_df['is_train'] == 0].reset_index(drop=True)
print(f"[*] Training samples: {len(train_df)}")
print(f"[*] Testing samples: {len(test_df)}")
train_dataset = CUBDataset(train_df, transform=train_transform, use_bounding_box=USE_BOUNDING_BOXES)
test_dataset = CUBDataset(test_df, transform=test_transform, use_bounding_box=USE_BOUNDING_BOXES)
if len(train_dataset) == 0 or len(test_dataset) == 0: raise ValueError("Datasets empty.")
num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 1)
print(f"[*] Using num_workers = {num_workers} for DataLoaders.")
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
print("[*] DataLoaders created.")

# --- Model Definition (remains the same) ---
print("[*] Initializing model (ResNet50 pre-trained)...")
weights = models.ResNet50_Weights.IMAGENET1K_V1; model = models.resnet50(weights=weights)
if FREEZE_PRETRAINED_WEIGHTS: print("[!] WARNING: FREEZE_PRETRAINED_WEIGHTS=True")
else: print("[*] FINE-TUNING: Unfreezing pre-trained layers.")
num_ftrs = model.fc.in_features; model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
print(f"    - Replaced final layer for {NUM_CLASSES} classes."); model = model.to(device)
print("[*] Model ready.")

# --- Loss Function and Optimizer (remains the same) ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_LR_STEP_SIZE, gamma=STEP_LR_GAMMA)
print(f"[*] Optimizer: Adam, LR: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
print(f"[*] Loss: CrossEntropyLoss")
print(f"[*] Scheduler: StepLR (step={STEP_LR_STEP_SIZE}, gamma={STEP_LR_GAMMA})")

# --- Training Loop ---
best_acc = 0.0

def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    global best_acc
    since = time.time()
    # Store history for plotting
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    epochs_completed = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # *** STORE LEARNING RATE ***
        current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else LEARNING_RATE
        history['lr'].append(current_lr)
        # *************************
        print(f"\n--- Epoch {epoch+1}/{num_epochs} --- LR: {current_lr:.1e} ---")

        # --- Training Phase ---
        model.train(); running_loss_train, running_corrects_train, total_samples_train = 0.0, 0, 0
        print("Starting Training Phase..."); phase_start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(); outputs = model(inputs); _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels); loss.backward(); optimizer.step()
            running_loss_train += loss.item() * inputs.size(0); running_corrects_train += torch.sum(preds == labels.data)
            total_samples_train += inputs.size(0)
        epoch_loss_train = running_loss_train / total_samples_train; epoch_acc_train = running_corrects_train.double() / total_samples_train
        history['train_loss'].append(epoch_loss_train); history['train_acc'].append(epoch_acc_train.item())
        print(f'Train Loss: {epoch_loss_train:.4f} Acc: {epoch_acc_train:.4f} Time: {time.time() - phase_start_time:.2f}s')

        # --- Validation Phase ---
        model.eval(); running_loss_val, running_corrects_val, total_samples_val = 0.0, 0, 0
        print("Starting Validation Phase..."); phase_start_time = time.time()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs); _, preds = torch.max(outputs, 1); loss = criterion(outputs, labels)
                running_loss_val += loss.item() * inputs.size(0); running_corrects_val += torch.sum(preds == labels.data)
                total_samples_val += inputs.size(0)
        epoch_loss_val = running_loss_val / total_samples_val; epoch_acc_val = running_corrects_val.double() / total_samples_val
        history['val_loss'].append(epoch_loss_val); history['val_acc'].append(epoch_acc_val.item())
        print(f'Valid Loss: {epoch_loss_val:.4f} Acc: {epoch_acc_val:.4f} Time: {time.time() - phase_start_time:.2f}s')

        # Save model if validation accuracy improves
        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val; print(f"[*] New best valid acc: {best_acc:.4f}. Saving model...")
            try: torch.save(model.state_dict(), MODEL_SAVE_NAME); print(f"[*] Model saved to '{MODEL_SAVE_NAME}'")
            except Exception as e: print(f"[!] Error saving model: {e}")

        scheduler.step(); epochs_completed += 1
        print(f"--- Epoch {epoch+1} Time: {time.time() - epoch_start_time:.2f}s ---")

    time_elapsed = time.time() - since
    print(f'\n[*] Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'[*] Best Validation Accuracy: {best_acc:4f}')

    # --- Plotting Section ---
    print("\n[*] Generating training history plots...")
    PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if epochs_completed > 0:
        epochs_range = range(1, epochs_completed + 1)

        # --- Plot Loss ---
        plt.figure(figsize=(12, 6)); plt.subplot(1, 2, 1) # Create figure with 2 subplots
        plt.plot(epochs_range, history['train_loss'], 'o-', label='Train Loss')
        plt.plot(epochs_range, history['val_loss'], 'o-', label='Validation Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss History'); plt.legend(); plt.grid(True)
        # --- Plot Accuracy ---
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_acc'], 'o-', label='Train Accuracy')
        plt.plot(epochs_range, history['val_acc'], 'o-', label='Validation Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy History'); plt.legend(); plt.grid(True)
        plt.tight_layout() # Adjust layout
        loss_acc_plot_path = PLOT_SAVE_DIR / f"{PLOT_LOSS_FILENAME.split('.')[0]}_{PLOT_ACC_FILENAME.split('.')[0]}.png"
        try: plt.savefig(loss_acc_plot_path); print(f"[*] Loss/Acc plot saved to: {loss_acc_plot_path}")
        except Exception as e: print(f"[!] Error saving Loss/Acc plot: {e}")
        plt.close()

        # --- Plot Learning Rate ---
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, history['lr'], 'o-')
        plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Schedule')
        plt.grid(True); plt.yscale('log') # Use log scale for LR usually
        lr_plot_path = PLOT_SAVE_DIR / PLOT_LR_FILENAME
        try: plt.savefig(lr_plot_path); print(f"[*] LR plot saved to: {lr_plot_path}")
        except Exception as e: print(f"[!] Error saving LR plot: {e}")
        plt.close()

        # --- Plot Training Class Distribution ---
        plt.figure(figsize=(12, 8))
        # Use seaborn if available for potentially better look
        if USE_SEABORN and not train_df.empty:
            sns.countplot(y=train_df['class_id'], order = train_df['class_id'].value_counts().index, palette='viridis')
            # Optional: Add class names to y-axis ticks if readable (might be too crowded for 200)
            # class_id_to_name_map = classes_df.set_index('class_id')['class_name'].to_dict()
            # plt.yticks(ticks=range(NUM_CLASSES), labels=[class_id_to_name_map.get(i+1, f'Class {i+1}') for i in range(NUM_CLASSES)]) # Careful with 0/1 indexing
        elif not train_df.empty:
             train_df['class_id'].value_counts().sort_index().plot(kind='barh') # Simple matplotlib bar chart
        else:
             plt.text(0.5, 0.5, "No training data found", ha='center', va='center')

        plt.xlabel('Number of Training Images'); plt.ylabel('Class ID')
        plt.title('Training Set Class Distribution')
        plt.gca().invert_yaxis() # Show most frequent classes at the top if using countplot order
        plt.tight_layout()
        class_dist_path = PLOT_SAVE_DIR / PLOT_CLASSDIST_FILENAME
        try: plt.savefig(class_dist_path); print(f"[*] Class Dist plot saved to: {class_dist_path}")
        except Exception as e: print(f"[!] Error saving Class Dist plot: {e}")
        plt.close()

        # --- Plot Bounding Box Area Distribution ---
        plt.figure(figsize=(10, 5))
        if not all_data_df.empty and all({'width', 'height'} <= set(all_data_df.columns)):
            bbox_areas = all_data_df['width'] * all_data_df['height']
            if USE_SEABORN:
                sns.histplot(bbox_areas, bins=50, kde=True)
            else:
                 plt.hist(bbox_areas, bins=50)
            plt.xlabel('Bounding Box Area (pixels^2)'); plt.ylabel('Frequency')
            plt.title('Distribution of Bounding Box Areas (All Images)')
            plt.grid(True)
        else:
             plt.text(0.5, 0.5, "Bounding box data missing", ha='center', va='center')

        plt.tight_layout()
        bbox_area_path = PLOT_SAVE_DIR / PLOT_BBOXAREA_FILENAME
        try: plt.savefig(bbox_area_path); print(f"[*] BBox Area plot saved to: {bbox_area_path}")
        except Exception as e: print(f"[!] Error saving BBox Area plot: {e}")
        plt.close()

    else:
        print("[*] No epochs completed, skipping plot generation.")

    return model

# --- Start Training ---
if __name__ == "__main__":
    print("[*] Starting fine-tuning process...")
    try:
        PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        trained_model = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)
        print("[*] Fine-tuning finished successfully.")
    except KeyboardInterrupt:
         print("\n[!] Training interrupted."); print(f"[*] Best valid acc achieved: {best_acc:.4f}")
    except Exception as e: print(f"\n[!!!] Error during training: {e}"); traceback.print_exc()
    finally: print("\n --- Script Finished ---")