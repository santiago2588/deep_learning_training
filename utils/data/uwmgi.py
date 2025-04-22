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
    
    Args:
        case_path (Path or str): Path to the case folder
    
    Returns:
        tuple: (list of image paths, list of image IDs)
    """
    img_paths = []
    img_ids = []
    
    # Convert to Path object if it's not already
    case_path = Path(case_path)
    
    # Get case number from folder name (case123 -> 123)
    case_num = case_path.name.replace("case", "")
    
    # Find all day folders
    day_folders = list(case_path.glob("*_day*"))
    
    for day_folder in day_folders:
        # Get day number from folder name (day20 -> 20)
        day_num = day_folder.name.split("_")[1].replace("day", "")
        
        # Find all scans in the day folder
        scan_files = list(day_folder.joinpath("scans").glob("*.png"))
        
        for scan_file in scan_files:
            # Get slice number from filename (slice_0001.png -> 0001)
            slice_num = scan_file.name.split("_")[1]
            
            # Format ID as case<num>_day<num>_slice_<slice_id>
            img_id = f"case{case_num}_day{day_num}_slice_{slice_num}"
            
            img_paths.append(scan_file)
            img_ids.append(img_id)
            
    return img_paths, img_ids

def load_img(img_path):
    """
    Load an image and convert from uint16 to uint8 format.
    
    Args:
        img_path (Path or str): Path to the image file
    
    Returns:
        numpy.ndarray: Loaded and converted image
    """
    # Convert to string for cv2
    img_path_str = str(img_path)
    
    # Read image as uint16
    img = cv2.imread(img_path_str, cv2.IMREAD_UNCHANGED)
    
    # Normalize and convert to uint8
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    
    return img

def rle_decode(rle_string, shape):
    """
    Decode RLE string to a binary mask.
    
    Args:
        rle_string (str): Run-length encoded string
        shape (tuple): Shape of the output mask (height, width)
    
    Returns:
        numpy.ndarray: Binary mask
    """
    if pd.isna(rle_string):
        return np.zeros(shape, dtype=np.uint8)
    
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # RLE is 1-indexed, convert to 0-indexed
    ends = starts + lengths
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    
    return mask.reshape(shape)

def rgb_to_onehot_to_gray(masks_dict, shape):
    """
    Convert RGB masks to grayscale with class IDs.
    
    Args:
        masks_dict (dict): Dictionary with class names as keys and binary masks as values
        shape (tuple): Shape of the output mask (height, width)
    
    Returns:
        numpy.ndarray: Grayscale mask with class IDs
    """
    # Initialize grayscale mask with zeros (background)
    gray_mask = np.zeros(shape, dtype=np.uint8)
    
    # Assign class IDs: 1 for 'large_bowel', 2 for 'small_bowel', 3 for 'stomach'
    class_ids = {
        'large_bowel': 1,
        'small_bowel': 2,
        'stomach': 3
    }
    
    # Fill in the grayscale mask with class IDs
    for class_name, binary_mask in masks_dict.items():
        if class_name in class_ids:
            gray_mask[binary_mask == 1] = class_ids[class_name]
    
    return gray_mask

def create_and_write_img_msk(img_paths, img_ids, df, output_dir, set_name):
    """
    Process images and masks and write them to the output directory.
    
    Args:
        img_paths (list): List of image paths
        img_ids (list): List of corresponding image IDs
        df (pandas.DataFrame): DataFrame with annotations
        output_dir (Path or str): Output directory
        set_name (str): Name of the set ('train' or 'val')
    """
    # Convert to Path object
    output_dir = Path(output_dir)
    
    # Create directories
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
        # Skip if image doesn't have annotations
        if img_id not in df['id'].values:
            skipped_count += 1
            continue
        
        # Load and convert image
        img = load_img(img_path)
        height, width = img.shape[:2]
        
        # Get annotations for this image
        img_df = df[df['id'] == img_id]
        
        # Create masks for each class
        masks_dict = {}
        for _, row in img_df.iterrows():
            if pd.notna(row['segmentation']):
                class_name = row['class']
                rle = row['segmentation']
                mask = rle_decode(rle, (height, width))
                masks_dict[class_name] = mask
        
        # Skip if no masks found
        if not masks_dict:
            skipped_count += 1
            continue
        
        # Convert masks to grayscale
        gray_mask = rgb_to_onehot_to_gray(masks_dict, (height, width))
        
        # Save image and mask
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
    
    Args:
        data_dir (Path or str, optional): Path to the raw dataset. If None, uses default path.
        output_dir (Path or str, optional): Path to save the processed dataset. If None, uses default path.
        train_ratio (float, optional): Ratio of training data. Default is 0.8.
        seed (int, optional): Random seed for reproducibility. Default is 42.
    
    Returns:
        Path: Path to the processed dataset
    """
    project_root = Path(find_project_root())
    
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
    
    # Read annotations CSV
    csv_path = data_dir.joinpath('train.csv')
    df = pd.read_csv(csv_path)
    
    # Get all case folders
    case_folders = list(data_dir.joinpath('train').glob("case*"))
    
    print(f"Found {len(case_folders)} case folders")
    
    # Collect all image paths and IDs
    all_img_paths = []
    all_img_ids = []
    
    # Show progress for collecting files from case folders
    for case_folder in tqdm(case_folders, desc="Collecting files from cases", unit="case"):
        img_paths, img_ids = get_folder_files(case_folder)
        all_img_paths.extend(img_paths)
        all_img_ids.extend(img_ids)
    
    print(f"Total images collected: {len(all_img_paths)}")
    
    # Create path-id pairs and filter for images with annotations
    path_id_pairs = list(zip(all_img_paths, all_img_ids))
    
    # Show progress for filtering annotated images
    with tqdm(desc="Filtering annotated images", total=len(path_id_pairs), unit="img") as pbar:
        filtered_pairs = []
        for path, img_id in path_id_pairs:
            has_annotation = img_id in df['id'].values
            if has_annotation:
                filtered_pairs.append((path, img_id))
            pbar.update(1)
    
    path_id_pairs = filtered_pairs
    print(f"Images with annotations: {len(path_id_pairs)}")
    
    # Shuffle and split into train/val sets
    np.random.seed(seed)
    np.random.shuffle(path_id_pairs)
    
    split_idx = int(len(path_id_pairs) * train_ratio)
    train_pairs = path_id_pairs[:split_idx]
    val_pairs = path_id_pairs[split_idx:]
    
    train_paths, train_ids = zip(*train_pairs) if train_pairs else ([], [])
    val_paths, val_ids = zip(*val_pairs) if val_pairs else ([], [])
    
    print(f"Split data into {len(train_paths)} training and {len(val_paths)} validation samples")
    
    # Process training set
    create_and_write_img_msk(train_paths, train_ids, df, output_dir, 'train')
    
    # Process validation set
    create_and_write_img_msk(val_paths, val_ids, df, output_dir, 'val')
    
    print(f"Preprocessing complete. Dataset saved to {output_dir}")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    return output_dir
