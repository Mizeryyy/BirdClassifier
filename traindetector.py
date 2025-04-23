# analyze_imgcount_vs_avgparts_hist.py
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

# Plotting & Analysis
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    USE_SEABORN = True
    print("[*] Using Seaborn for plotting.")
except ImportError:
    USE_SEABORN = False
    print("[!] Seaborn not found. Plots will use default matplotlib style.")

# --- Configuration ---
DATASET_BASE_DIR = Path("./")
PLOT_SAVE_DIR = Path("./dataset_analysis_plots") # Directory to save plots
PLOT_FILENAME = "species_img_count_vs_avg_parts_hist2d.png" # New filename

# --- Setup ---
if (DATASET_BASE_DIR / "CUB_200_2011").is_dir():
    DATASET_PATH = DATASET_BASE_DIR / "CUB_200_2011"
elif DATASET_BASE_DIR.name == "CUB_200_2011" and DATASET_BASE_DIR.is_dir():
     DATASET_PATH = DATASET_BASE_DIR
else:
    raise FileNotFoundError(f"CUB_200_2011 dataset not found in {DATASET_BASE_DIR}")

print(f"[*] Using dataset path: {DATASET_PATH}")

# --- Data Loading Function ---
def load_species_part_image_data(dataset_path):
    """Loads and merges data needed for this analysis."""
    # (Implementation is identical to the previous script)
    labels_file = dataset_path / 'image_class_labels.txt'
    classes_file = dataset_path / 'classes.txt'
    parts_loc_file = dataset_path / 'parts/part_locs.txt'
    print(f"[*] Loading data for image count vs. average parts analysis...")
    data_frames = {}; all_files_found = True
    files_to_load = {
        'labels': (labels_file, ['img_id', 'class_id']),
        'classes': (classes_file, ['class_id', 'class_name']),
        'part_locs': (parts_loc_file, ['img_id', 'part_id', 'x', 'y', 'visible'])
    }
    for key, (path, names) in files_to_load.items():
        try:
            if path.exists():
                sep = r'\s+' if key in ['classes', 'part_locs'] else ' '
                data_frames[key] = pd.read_csv(path, sep=sep, names=names, header=None, on_bad_lines='warn')
                print(f"    - Loaded {path.name} ({len(data_frames[key])} rows)")
            else:
                if key == 'part_locs': print(f"[!] Error: Part locations file not found: {path}."); all_files_found = False; data_frames[key] = None
                else: print(f"[!] Error: Required file not found: {path}"); all_files_found = False; data_frames[key] = None
        except Exception as e: print(f"[!] Error loading {path.name}: {e}"); all_files_found = False; data_frames[key] = None
    if not all_files_found: raise RuntimeError("Could not load all required metadata files.")

    print("[*] Processing and merging data...")
    try:
        classes_df = data_frames['classes']; classes_df['class_name'] = classes_df['class_name'].str.split('.').str[1]
        part_locs_df = data_frames['part_locs']; part_locs_df['visible'] = pd.to_numeric(part_locs_df['visible'], errors='coerce')
        part_locs_df.dropna(subset=['visible'], inplace=True); part_locs_df['visible'] = part_locs_df['visible'].astype(int)
        visible_parts = part_locs_df[part_locs_df['visible'] == 1]
        part_counts = visible_parts.groupby('img_id').size().reset_index(name='num_parts_visible')
        labels_classes_df = pd.merge(data_frames['labels'], classes_df, on='class_id', how='inner')
        merged_df = pd.merge(labels_classes_df, part_counts, on='img_id', how='left')
        merged_df['num_parts_visible'].fillna(0, inplace=True); merged_df['num_parts_visible'] = merged_df['num_parts_visible'].astype(int)
        print(f"[*] Data merged successfully. Total entries with labels: {len(merged_df)}")
        return merged_df[['class_name', 'img_id', 'num_parts_visible']]
    except Exception as e: print(f"[!] An unexpected error occurred during data processing: {e}"); traceback.print_exc(); return None


# --- Plotting Function (Modified for 2D Histogram) ---
def plot_2d_histogram(data_df, x_col, y_col, title, xlabel, ylabel, filename, directory, x_bins=20, y_bins=20):
    """Generates and saves a 2D histogram (heatmap)."""
    if data_df is None or not all(c in data_df.columns for c in [x_col, y_col]):
        print(f"[!] Cannot generate 2D histogram '{title}': Data or columns missing.")
        return

    plot_data = data_df[[x_col, y_col]].dropna()
    if plot_data.empty:
         print(f"[!] Cannot generate 2D histogram '{title}': No valid data points after dropping NaNs.")
         return

    print(f"[*] Generating 2D histogram: {title}...")
    plt.figure(figsize=(10, 7))

    try:
        if USE_SEABORN:
            # Use histplot with both x and y for a 2D histogram heatmap
            sns.histplot(
                data=plot_data,
                x=x_col,
                y=y_col,
                bins=(x_bins, y_bins), # Specify bins for both axes
                cbar=True,       # Add a color bar indicating counts
                cmap='viridis'   # Choose a colormap (e.g., 'viridis', 'magma', 'plasma')
            )
        else:
            # Matplotlib fallback using hist2d
            print("[!] Seaborn not available. Generating basic 2D histogram with Matplotlib.")
            counts, xedges, yedges, im = plt.hist2d(
                x=plot_data[x_col],
                y=plot_data[y_col],
                bins=(x_bins, y_bins),
                cmap='viridis'
            )
            plt.colorbar(im, label='Number of Species') # Add color bar

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Optional: Adjust grid, maybe remove it for heatmaps
        # plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Save the plot
        filepath = directory / filename
        plt.savefig(filepath, dpi=120)
        print(f"[*] Plot saved successfully to: {filepath}")

    except Exception as e:
        print(f"[!] Error generating 2D histogram {filename}: {e}")
        traceback.print_exc()
    finally:
        plt.close() # Ensure plot is closed


# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Analyzing Image Count vs. Average Visible Parts per Species (2D Hist) ---")
    PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Load and process the data
        merged_data = load_species_part_image_data(DATASET_PATH)

        if merged_data is not None:
            # Aggregate per Species
            print("[*] Aggregating data per species...")
            species_stats = merged_data.groupby('class_name').agg(
                image_count=('img_id', 'size'), # Count images per species
                avg_visible_parts=('num_parts_visible', 'mean') # Avg visible parts
            ).reset_index()

            if species_stats.empty:
                raise RuntimeError("Aggregation resulted in empty dataframe.")

            print(f"    Aggregated stats calculated for {len(species_stats)} species.")

            # Determine reasonable bin counts
            x_bin_count = min(25, max(10, len(species_stats['image_count'].unique()) // 2)) # Adjust based on unique image counts
            y_bin_count = 20 # Number of bins for average parts

            # Generate the 2D histogram plot
            plot_2d_histogram(
                data_df=species_stats,
                x_col='image_count',
                y_col='avg_visible_parts',
                title='Density of Species: Image Count vs. Average Visible Parts',
                xlabel='Number of Images in Dataset per Species',
                ylabel='Average Number of Visible Parts Annotated per Species',
                filename=PLOT_FILENAME,
                directory=PLOT_SAVE_DIR,
                x_bins = x_bin_count,
                y_bins = y_bin_count
            )

            # Correlation calculation is removed as requested (less direct for histograms)

        else:
            print("[!] Failed to load or process data needed for plotting.")

    except Exception as e:
        print(f"\n[!!!] An error occurred during the analysis script: {e}")
        traceback.print_exc()
    finally:
        print("\n --- Image Count vs. Parts 2D Histogram Analysis Script Finished ---")