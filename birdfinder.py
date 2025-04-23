# predict_birds.py
import torch
import torchvision
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import pandas as pd
import os
from pathlib import Path
import time
import traceback

# --- Configuration ---

# --- Paths ---
# *** Path to the FINE-TUNED model weights file saved by main.py ***
MODEL_PATH = Path("./cub_bird_classifier_resnet50_finetuned.pth")

# Path to the classes.txt file from the CUB dataset folder
CLASSES_TXT_PATH = Path("./CUB_200_2011/classes.txt")
# Folder containing the NEW bird images you want to classify
INPUT_IMAGE_DIR = Path("./test_images") # ! MODIFY IF NEEDED (Folder with your test images)
# Folder where annotated images WILL BE SAVED
OUTPUT_DIR = Path("./annotated_output") # ! MODIFY IF NEEDED (Output folder)

# --- Model & Training Parameters (MUST MATCH THE TRAINING SCRIPT) ---
NUM_CLASSES = 200
INPUT_SIZE = 224 # Should match INPUT_SIZE from main.py
MODEL_ARCHITECTURE = models.resnet50 # Should match the model architecture from main.py

# --- Inference Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Annotation Appearance ---
# Optional: Provide a path to a .ttf font file for better text rendering
# FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # Example for Linux
# FONT_PATH = "C:/Windows/Fonts/Arial.ttf" # Example for Windows
FONT_PATH = None # Set to None to try Pillow's default font
FONT_SIZE = 24
TEXT_COLOR = "yellow"
BOX_COLOR = "black" # Background box for text

# --- Helper Functions ---

def load_class_names(filepath):
    """Loads class names from classes.txt"""
    try:
        classes_df = pd.read_csv(filepath, sep=r'\s+', names=['class_id', 'class_name'], header=None)
        # CUB classes are 1-indexed in the file, map to 0-indexed used in training/prediction
        class_id_to_name = {i: name for i, name in enumerate(classes_df['class_name'])}
        print(f"[*] Loaded {len(class_id_to_name)} class names from {filepath}.")
        return class_id_to_name
    except FileNotFoundError:
        print(f"[!] Error: classes.txt not found at {filepath}")
        return None
    except Exception as e:
        print(f"[!] Error loading or parsing {filepath}: {e}")
        return None

def load_trained_model(model_path, model_architecture, num_classes, device):
    """Loads the specified model structure and trained weights."""
    print(f"[*] Attempting to load model architecture: {model_architecture.__name__}")
    try:
        # Initialize the model structure (weights=None avoids using default pre-trained weights)
        model = model_architecture(weights=None)
    except Exception as e:
        print(f"[!] Failed to initialize model architecture {model_architecture.__name__}: {e}")
        return None

    # Modify the final layer to match the number of classes *exactly* as done in training
    try:
        if isinstance(model, models.ResNet):
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
            print(f"    - Replaced ResNet final layer 'fc' for {num_classes} classes.")
        # Add elif blocks here for other architectures if needed (e.g., EfficientNet, ViT)
        else:
             print(f"[!] Warning: Final layer replacement logic not implemented for {type(model)}. Assuming structure in saved weights matches.")
    except AttributeError as e:
         print(f"[!] Warning: Could not find standard final layer (e.g., 'fc') for {type(model)}: {e}. Assuming structure in saved weights matches.")

    print(f"[*] Loading trained weights from: {model_path}")
    if not model_path.exists():
        print(f"[!] Error: Model weights file not found at {model_path}")
        print("[!] Please ensure the training script (`main.py`) ran successfully and saved the model.")
        return None

    try:
        # Load the saved state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("[*] Weights loaded successfully.")
    except RuntimeError as e:
         print(f"[!] Error loading model weights: {e}")
         print("[!] This often means the model architecture defined here (or its final layer)")
         print("[!] does not precisely match the one used when the weights were saved.")
         return None
    except Exception as e:
        print(f"[!] An unexpected error occurred loading model weights: {e}")
        return None

    model.to(device)
    model.eval()  # Set model to evaluation mode (VERY IMPORTANT!)
    print(f"[*] Model ready for inference on device: {device}.")
    return model

# --- Define Transformations (MUST match validation/test transforms from main.py) ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

inference_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)), # Resize first
    transforms.CenterCrop(INPUT_SIZE),           # Then crop center
    transforms.ToTensor(),
    normalize,
])
print(f"[*] Using inference transforms (Resize/CenterCrop: {INPUT_SIZE}x{INPUT_SIZE})")


# --- Main Inference Script ---
if __name__ == "__main__":
    print("\n--- Bird Species Prediction Script ---")

    # --- Preparations ---
    class_names = load_class_names(CLASSES_TXT_PATH)
    if class_names is None: exit(1)

    model = load_trained_model(MODEL_PATH, MODEL_ARCHITECTURE, NUM_CLASSES, DEVICE)
    if model is None: exit(1)

    if not INPUT_IMAGE_DIR.is_dir():
        print(f"[!] Error: Input image directory not found: {INPUT_IMAGE_DIR}")
        exit(1)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'] # Add more if needed
    image_files = []
    for ext in image_extensions:
        image_files.extend(INPUT_IMAGE_DIR.glob(f'*{ext}'))
        image_files.extend(INPUT_IMAGE_DIR.glob(f'*{ext.upper()}'))

    image_files = sorted(list(set(image_files))) # Unique sorted list

    if not image_files:
        print(f"[!] No images found with extensions {image_extensions} in {INPUT_IMAGE_DIR}")
        exit(1)
    print(f"[*] Found {len(image_files)} images to process in {INPUT_IMAGE_DIR}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Annotated images will be saved to: {OUTPUT_DIR}")

    # Load font
    loaded_font = None
    try:
        if FONT_PATH and Path(FONT_PATH).exists():
             loaded_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
             print(f"[*] Using font: {FONT_PATH}")
        else:
             loaded_font = ImageFont.load_default()
             print("[*] Using default PIL font.")
    except Exception as e:
        print(f"[!] Warning: Could not load font. Text drawing might fail. Error: {e}")


    # --- Process Each Image ---
    start_time_all = time.time()
    predictions_summary = []
    processed_count = 0

    for i, img_path in enumerate(image_files):
        print(f"\n--- Processing [{i+1}/{len(image_files)}]: {img_path.name} ---")
        start_time_img = time.time()
        predicted_class_name = "Error"
        confidence = 0.0

        try:
            # Load the original image using context manager
            with Image.open(img_path) as original_img:
                # Ensure image is in RGB for consistency
                img_rgb = original_img.convert('RGB')
                # Apply transformations to a copy for the model input
                img_transformed = inference_transform(img_rgb)

            # --- Prepare for Model ---
            input_batch = img_transformed.unsqueeze(0) # Add batch dimension: [C, H, W] -> [1, C, H, W]
            input_batch = input_batch.to(DEVICE)

            # --- Perform Inference ---
            with torch.no_grad(): # Disable gradient calculations
                output = model(input_batch)

            # --- Get Prediction ---
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence_tensor, predicted_idx_tensor = torch.max(probabilities, dim=0)

            predicted_idx = predicted_idx_tensor.item()
            confidence = confidence_tensor.item()

            # Map index to class name (handle potential unknown index)
            predicted_class_name = class_names.get(predicted_idx, f"Unknown Class ID: {predicted_idx}")

            print(f"[*] Prediction: '{predicted_class_name}' (Index: {predicted_idx})")
            print(f"[*] Confidence: {confidence:.4f}")
            predictions_summary.append({'filename': img_path.name, 'prediction': predicted_class_name, 'confidence': confidence})

            # --- Annotate Original Image ---
            # Re-open the original image to draw on it (ensures we have the original dimensions/quality)
            with Image.open(img_path) as img_to_annotate:
                 img_to_annotate = img_to_annotate.convert('RGB') # Ensure RGB again for drawing
                 draw = ImageDraw.Draw(img_to_annotate)
                 text = f"{predicted_class_name} ({confidence:.2f})" # Text to draw

                 if loaded_font:
                     try:
                         # Calculate text bounding box for background rectangle
                         if hasattr(draw, 'textbbox'): # Modern Pillow
                             text_bbox = draw.textbbox((0, 0), text, font=loaded_font)
                         else: # Older Pillow fallback
                             text_w, text_h = draw.textsize(text, font=loaded_font)
                             text_bbox = (0, 0, text_w, text_h)

                         text_width = text_bbox[2] - text_bbox[0]
                         text_height = text_bbox[3] - text_bbox[1]

                         margin = int(FONT_SIZE * 0.2) # Small margin around text
                         rect_left = margin
                         rect_top = margin
                         rect_right = rect_left + text_width + margin * 2
                         rect_bottom = rect_top + text_height + margin * 2

                         # Draw background rectangle
                         draw.rectangle((rect_left, rect_top, rect_right, rect_bottom), fill=BOX_COLOR)
                         # Draw text (position slightly inside the rectangle)
                         text_x = rect_left + margin
                         text_y = rect_top + margin
                         draw.text((text_x, text_y), text, fill=TEXT_COLOR, font=loaded_font)

                     except Exception as e:
                         print(f"[!] Warning: Error during text drawing: {e}")
                 else:
                     print("[!] Skipping text annotation due to font loading issue.")

                 # --- Save Annotated Image ---
                 # Prepend "predicted_" to the original filename
                 output_filename = OUTPUT_DIR / f"predicted_{img_path.name}"
                 try:
                    img_to_annotate.save(output_filename)
                    print(f"[*] Annotated image saved to: {output_filename}")
                    processed_count += 1
                 except Exception as e:
                    print(f"[!] Error saving annotated image {output_filename}: {e}")

        except FileNotFoundError:
            print(f"[!] Error: Input image file not found at {img_path}")
            predictions_summary.append({'filename': img_path.name, 'prediction': 'File Not Found', 'confidence': 0.0})
        except UnidentifiedImageError:
             print(f"[!] Error: Could not read image file (corrupted/unsupported format?): {img_path}.")
             predictions_summary.append({'filename': img_path.name, 'prediction': 'Corrupted Image', 'confidence': 0.0})
        except Exception as e:
            print(f"[!] An unexpected error occurred processing {img_path.name}: {e}")
            traceback.print_exc() # Print detailed error for debugging
            predictions_summary.append({'filename': img_path.name, 'prediction': 'Processing Error', 'confidence': 0.0})

        elapsed_img = time.time() - start_time_img
        print(f"[*] Time taken: {elapsed_img:.2f}s")


    # --- Finish ---
    total_elapsed = time.time() - start_time_all
    print(f"\n--- Prediction finished for {processed_count}/{len(image_files)} images ---")
    if image_files:
         print(f"[*] Total time: {total_elapsed:.2f}s")
         print(f"[*] Average time per image: {total_elapsed / len(image_files):.2f}s")
    print(f"[*] Annotated images are saved in: {OUTPUT_DIR}")

    # Optional: Save summary to a CSV file in the output directory
    if predictions_summary:
        try:
            summary_df = pd.DataFrame(predictions_summary)
            summary_csv_path = OUTPUT_DIR / "_predictions_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"[*] Prediction summary saved to: {summary_csv_path}")
        except Exception as e:
            print(f"[!] Could not save prediction summary CSV: {e}")

    print("\n --- Script Finished ---")