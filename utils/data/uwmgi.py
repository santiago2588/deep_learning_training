from pathlib import Path
from tqdm import tqdm
from utils.core import find_project_root
import numpy as np
import pandas as pd
import cv2


__all__ = ["preprocess_uwmgi_dataset"]

def get_folder_files(case_path):
    """
    Get all image files and their corresponding IDs from a case folder.
    
    This function traverses the directory structure of a case folder in the
    UW-Madison GI Tract Image Segmentation dataset and extracts all image paths
    and their corresponding identifiers.
    
    Args:
        case_path (Path or str): Path to the case folder
    
    Returns:
        tuple: (list of image paths, list of image IDs)
            - Image paths are Path objects pointing to the scan files
            - Image IDs are formatted as case<num>_day<num>_slice_<slice_id>
    """
    img_paths = []
    img_ids = []
    
    # Convert to Path object if it's not already
    case_path = Path(case_path)
    
    # Get case number from folder name (case123 -> 123)
    case_num = case_path.name.replace("case", "")
    
    # Find all day folders (e.g., patient_1_day0, patient_1_day20)
    day_folders = list(case_path.glob("*_day*"))
    
    for day_folder in day_folders:
        # Get day number from folder name (day20 -> 20)
        day_num = day_folder.name.split("_")[1].replace("day", "")
        
        # Find all scan PNG files in the scans subfolder
        scan_files = list(day_folder.joinpath("scans").glob("*.png"))
        
        for scan_file in scan_files:
            # Get slice number from filename (slice_0001.png -> 0001)
            slice_num = scan_file.name.split("_")[1]
            
            # Format ID as case<num>_day<num>_slice_<slice_id>
            # This matches the ID format in the annotations CSV
            img_id = f"case{case_num}_day{day_num}_slice_{slice_num}"
            
            img_paths.append(scan_file)
            img_ids.append(img_id)
            
    return img_paths, img_ids

def load_img(img_path):
    """
    Load an image and convert from uint16 to uint8 format.
    
    This function reads a medical image and normalizes it to the uint8 range 
    for easier processing and visualization.
    
    Args:
        img_path (Path or str): Path to the image file
    
    Returns:
        numpy.ndarray: Loaded and normalized image in uint8 format (0-255)
    """
    # Convert to string for cv2 compatibility
    img_path_str = str(img_path)
    
    # Read image as uint16 (16-bit depth)
    img = cv2.imread(img_path_str, cv2.IMREAD_UNCHANGED)
    
    # Normalize to 0-255 range and convert to uint8 for standard image processing
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    
    return img

def rle_decode(rle_string, shape):
    """
    Decode RLE (Run-Length Encoding) string to a binary mask.
    
    Run-Length Encoding is a common format for storing segmentation masks.
    This function converts an RLE string to a binary mask of specified shape.
    
    Args:
        rle_string (str): Run-length encoded string where pairs of values indicate
                         (start position, run length)
        shape (tuple): Shape of the output mask (height, width)
    
    Returns:
        numpy.ndarray: Binary mask where 1 indicates the segmented region
    """
    # Handle empty masks
    if pd.isna(rle_string):
        return np.zeros(shape, dtype=np.uint8)
    
    # Parse the RLE string into starts and lengths
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    
    # RLE is 1-indexed in the dataset format, convert to 0-indexed for Python
    starts -= 1
    
    # Calculate end positions for each run
    ends = starts + lengths
    
    # Create flattened mask and set run regions to 1
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    
    # Reshape to specified dimensions
    return mask.reshape(shape)

def rgb_to_onehot_to_gray(masks_dict, shape):
    """
    Convert class-specific binary masks to a single grayscale mask with class IDs.
    
    This function takes separate binary masks for each class and combines them
    into a single grayscale image where pixel values represent class IDs.
    
    Args:
        masks_dict (dict): Dictionary with class names as keys and binary masks as values
        shape (tuple): Shape of the output mask (height, width)
    
    Returns:
        numpy.ndarray: Grayscale mask where pixel values represent class IDs:
                      0 = background
                      1 = large bowel
                      2 = small bowel
                      3 = stomach
    """
    # Initialize grayscale mask with zeros (background)
    gray_mask = np.zeros(shape, dtype=np.uint8)
    
    # Assign class IDs based on the segmentation class
    class_ids = {
        'large_bowel': 1,
        'small_bowel': 2,
        'stomach': 3
    }
    
    # Fill in the grayscale mask with class IDs
    # If pixels overlap between classes, the last class processed takes precedence
    for class_name, binary_mask in masks_dict.items():
        if class_name in class_ids:
            gray_mask[binary_mask == 1] = class_ids[class_name]
    
    return gray_mask

def create_and_write_img_msk(img_paths, img_ids, df, output_dir, set_name):
    """
    Process images and masks and write them to the output directory.
    
    This function:
    1. Creates output directories for images and masks
    2. Processes and writes images and their corresponding masks 
    3. Tracks progress and counts of processed/skipped images
    
    Args:
        img_paths (list): List of image paths
        img_ids (list): List of corresponding image IDs
        df (pandas.DataFrame): DataFrame with annotations
        output_dir (Path or str): Output directory base path
        set_name (str): Name of the set ('train' or 'val')
    """
    # Convert to Path object for consistent path handling
    output_dir = Path(output_dir)
    
    # Create output directories for images and masks
    images_dir = output_dir.joinpath(set_name, 'images')
    masks_dir = output_dir.joinpath(set_name, 'masks')
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Progress bar for processing images and masks
    progress_bar = tqdm(zip(img_paths, img_ids), 
                        total=len(img_paths), 
                        desc=f"Processing {set_name} set", 
                        unit="img")
    
    processed_count = 0
    skipped_count = 0
    
    for img_path, img_id in progress_bar:
        # Skip if image doesn't have annotations in the dataframe
        if img_id not in df['id'].values:
            skipped_count += 1
            continue
        
        # Load and convert image
        img = load_img(img_path)
        height, width = img.shape[:2]
        
        # Get annotations for this image only
        img_df = df[df['id'] == img_id]
        
        # Create masks for each class (large_bowel, small_bowel, stomach)
        masks_dict = {}
        for _, row in img_df.iterrows():
            if pd.notna(row['segmentation']):
                class_name = row['class']
                rle = row['segmentation']
                mask = rle_decode(rle, (height, width))
                masks_dict[class_name] = mask
        
        # Skip if no valid masks found
        if not masks_dict:
            skipped_count += 1
            continue
        
        # Convert multiple binary masks to a single grayscale mask with class IDs
        gray_mask = rgb_to_onehot_to_gray(masks_dict, (height, width))
        
        # Save image and mask with the same filename for easy pairing
        img_filename = f"{img_id}.png"
        mask_filename = f"{img_id}.png"
        
        cv2.imwrite(str(images_dir.joinpath(img_filename)), img)
        cv2.imwrite(str(masks_dir.joinpath(mask_filename)), gray_mask)
        
        processed_count += 1
        
        # Update progress bar description with processed counts
        progress_bar.set_postfix(processed=processed_count, skipped=skipped_count)

def preprocess_uwmgi_dataset(data_dir=None, output_dir=None, train_ratio=0.8, seed=42):
    """
    Preprocess the UW-Madison GI Tract Image Segmentation dataset.
    
    This function:
    1. Loads the raw dataset and annotations
    2. Extracts all image files and their corresponding IDs
    3. Filters images to keep only those with annotations
    4. Splits the dataset into training and validation sets
    5. Processes and saves images and their segmentation masks
    
    Args:
        data_dir (Path or str, optional): Path to the raw dataset. If None, uses default path.
        output_dir (Path or str, optional): Path to save the processed dataset. If None, uses default path.
        train_ratio (float, optional): Ratio of training data. Default is 0.8 (80% train, 20% validation).
        seed (int, optional): Random seed for reproducibility. Default is 42.
    
    Returns:
        Path: Path to the processed dataset
    """
    # Get project root for default paths
    project_root = Path(find_project_root())
    
    # Set default paths if not provided
    if data_dir is None:
        data_dir = project_root.joinpath('data', 'raw', 'uwmgi')
    else:
        data_dir = Path(data_dir)
    
    if output_dir is None:
        output_dir = project_root.joinpath('data', 'processed', 'uwmgi')
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read annotations CSV containing RLE-encoded masks
    csv_path = data_dir.joinpath('train.csv')
    df = pd.read_csv(csv_path)
    
    # Get all case folders from the train directory
    case_folders = list(data_dir.joinpath('train').glob("case*"))
    
    print(f"Found {len(case_folders)} case folders")
    
    # Collect all image paths and IDs
    all_img_paths = []
    all_img_ids = []
    
    # Show progress while collecting files from case folders
    for case_folder in tqdm(case_folders, desc="Collecting files from cases", unit="case"):
        img_paths, img_ids = get_folder_files(case_folder)
        all_img_paths.extend(img_paths)
        all_img_ids.extend(img_ids)
    
    print(f"Total images collected: {len(all_img_paths)}")
    
    # Create path-id pairs and filter for images with annotations
    path_id_pairs = list(zip(all_img_paths, all_img_ids))
    
    # Show progress while filtering for annotated images
    with tqdm(desc="Filtering annotated images", total=len(path_id_pairs), unit="img") as pbar:
        filtered_pairs = []
        for path, img_id in path_id_pairs:
            has_annotation = img_id in df['id'].values
            if has_annotation:
                filtered_pairs.append((path, img_id))
            pbar.update(1)
    
    path_id_pairs = filtered_pairs
    print(f"Images with annotations: {len(path_id_pairs)}")
    
    # Shuffle and split into train/val sets with fixed random seed for reproducibility
    np.random.seed(seed)
    np.random.shuffle(path_id_pairs)
    
    split_idx = int(len(path_id_pairs) * train_ratio)
    train_pairs = path_id_pairs[:split_idx]
    val_pairs = path_id_pairs[split_idx:]
    
    # Handle empty sets gracefully
    train_paths, train_ids = zip(*train_pairs) if train_pairs else ([], [])
    val_paths, val_ids = zip(*val_pairs) if val_pairs else ([], [])
    
    print(f"Split data into {len(train_paths)} training and {len(val_paths)} validation samples")
    
    # Process and save training set
    create_and_write_img_msk(train_paths, train_ids, df, output_dir, 'train')
    
    # Process and save validation set
    create_and_write_img_msk(val_paths, val_ids, df, output_dir, 'val')
    
    print(f"Preprocessing complete. Dataset saved to {output_dir}")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    return output_dir
