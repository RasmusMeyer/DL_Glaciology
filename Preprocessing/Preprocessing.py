import os
import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import Window

## These functions were written partially by generative AI under careful monitoring. 

def merge_features_and_labels(features_path, labels_path, training_data_path):

    # Use rasterio to open raster satellite images and read bands (channels)
    with rasterio.open(features_path) as features_src:
        features_data = features_src.read()  #Shape - (Channels, H, W)

    #Labels corresponding to the satellite image
    with rasterio.open(labels_path) as labels_src:
        labels_data = labels_src.read(1)  #Shape - (Channels, H, W)

    if features_data.shape[1:] != labels_data.shape:
        print(f"Dimensions mismatch: {features_path} and {labels_path}. Skipping.")
        return
    
    merged_data = np.concatenate((features_data, labels_data[np.newaxis, :, :]), axis=0)

    # Save the merged data to a new file
    with rasterio.open(
        training_data_path, 'w', driver='GTiff',
        count=merged_data.shape[0], 
        dtype=merged_data.dtype,
        width=merged_data.shape[2], 
        height=merged_data.shape[1],
        crs=features_src.crs, transform=features_src.transform
    ) as dst:
        dst.write(merged_data)

    print(f"Saved merged file: {training_data_path}")


def split_into_patches_with_overlap_percentage(image_path, patch_size=256, overlap_percentage=0.25, num_features=None):
    """
    Split an image into overlapping patches with overlap specified as a percentage of the patch size.
    
    Args:
        image_path (str): Path to the image (features + labels combined).
        patch_size (int): Size of each patch (assumes square patches).
        overlap_percentage (float): Fraction of patch size to overlap (0 <= overlap_percentage < 1).
        num_features (int): Number of feature bands (None uses all bands except the last for features).
        
    Returns:
        patches_features (list): List of feature patches.
        patches_labels (list): List of label patches.
    """
    with rasterio.open(image_path) as src:
        # Get image dimensions and number of bands
        width, height = src.width, src.height
        num_bands = src.count
        
        # Ensure valid num_features is provided
        if num_features is None:
            num_features = num_bands - 1  # Default to all but the last band as features
        
        # Ensure that the last band is used for labels
        assert num_features < num_bands, "Number of features must be less than total number of bands."
        
        patches_features = []
        patches_labels = []
        
        # Calculate step size based on overlap percentage
        step = int(patch_size * (1 - overlap_percentage))
        if step <= 0:
            raise ValueError("Overlap percentage is too high; it results in zero or negative step size.")
        
        for top in range(0, height, step):
            for left in range(0, width, step):
                # Determine patch dimensions
                bottom = min(top + patch_size, height)
                right = min(left + patch_size, width)
                
                # Adjust for edge patches
                patch_height = bottom - top
                patch_width = right - left
                
                # Create a window for the feature patch (all bands except the last)
                window_features = Window(left, top, patch_width, patch_height)
                patch_features = src.read(range(1, num_features + 1), window=window_features)
                patch_features = np.moveaxis(patch_features, 0, -1)  # Convert to (H, W, C)
                
                # Create a window for the label patch (last band)
                window_labels = Window(left, top, patch_width, patch_height)
                patch_labels = src.read(num_features + 1, window=window_labels)
                patch_labels = patch_labels.astype(np.uint8)  # Ensure label patch is in uint8
                
                # Pad patches if they are smaller than patch_size
                if patch_height < patch_size or patch_width < patch_size:
                    padding_features = ((0, patch_size - patch_height), 
                                        (0, patch_size - patch_width), 
                                        (0, 0))
                    patch_features = np.pad(patch_features, padding_features, mode='constant', constant_values=0)
                    
                    padding_labels = ((0, patch_size - patch_height), 
                                      (0, patch_size - patch_width))
                    patch_labels = np.pad(patch_labels, padding_labels, mode='constant', constant_values=0)
                
                patches_features.append(patch_features)
                patches_labels.append(patch_labels)
        
        return patches_features, patches_labels
    

def get_patches(training_data_folder, patch_size=256, overlap_percentage=0.5, num_features=7):
    """
    Process all .tif files in the training data folder to extract patches.
    
    Args:
        training_data_folder (str): Path to the folder containing .tif image files.
        patch_size (int): Size of each patch (assumes square patches).
        overlap_percentage (float): Fraction of patch size to overlap (0 <= overlap_percentage < 1).
        num_features (int): Number of image bands to be used as features.
        
    Returns:
        all_patches_features (list): List of all feature patches.
        all_patches_labels (list): List of all label patches.
    """
    # List all .tif files in the folder
    image_files = [f for f in os.listdir(training_data_folder) if f.endswith('.tif')]
    
    all_patches_features = []
    all_patches_labels = []
    
    # Loop through the image files and process them
    for image_file in image_files:
        image_path = os.path.join(training_data_folder, image_file)
        
        patches_features, patches_labels = split_into_patches_with_overlap_percentage(
            image_path, patch_size=patch_size, overlap_percentage=overlap_percentage, num_features=num_features
        )
        
        all_patches_features.extend(patches_features)
        all_patches_labels.extend(patches_labels)
    
    print(f"Total Feature patches: {len(all_patches_features)}")
    print(f"Total Label patches: {len(all_patches_labels)}")
    
    return all_patches_features, all_patches_labels


import random
import torch
from torch.utils.data import Dataset, DataLoader


def filter_patches(patches, label_patches, min_classes=2):
    """
    Filter out patches that don't contain at least `min_classes` unique classes in the label patch.

    Args:
        patches (list of torch.Tensor): List of 4-band image patches.
        label_patches (list of torch.Tensor): List of label patches.
        min_classes (int): Minimum number of unique classes required in the label.

    Returns:
        filtered_patches (list of torch.Tensor): Filtered image patches.
        filtered_labels (list of torch.Tensor): Filtered label patches.
    """
    filtered_patches = []
    filtered_labels = []
    for patch, label_patch in zip(patches, label_patches):
        label_patch_tensor = torch.tensor(label_patch, dtype=torch.long)
        unique_classes = torch.unique(label_patch_tensor)
        
        if len(unique_classes) >= min_classes:
            filtered_patches.append(patch)
            filtered_labels.append(label_patch)
    
    return filtered_patches, filtered_labels


class PatchSegmentationDataset(Dataset):
    def __init__(self, patches, label_patches, num_classes):
        """
        Dataset to load image patches and their corresponding labels.

        Args:
            patches (list of torch.Tensor): List of image patches.
            label_patches (list of torch.Tensor): List of label patches.
            num_classes (int): Number of classes for segmentation.
        """
        self.patches = patches
        self.label_patches = label_patches
        self.num_classes = num_classes

    def __len__(self):
        """Return the number of samples (patches)."""
        return len(self.patches)

    def __getitem__(self, idx):
        """Return the image patch and its corresponding label patch."""
        # Ensure the image has the correct shape: (C, H, W)
        image = self.patches[idx].clone().detach().float()#.permute(2, 0, 1)  # (7, 256, 256)
        #mask = torch.tensor(self.label_patches[idx], dtype=torch.long)
        mask = self.label_patches[idx].clone().detach().long()
        # Replace NaN values in the image and mask with 0
        image = torch.nan_to_num(image, nan=0.0)
        mask = torch.nan_to_num(mask, nan=0)

        # Adjust range 
        image = image / 50000.0 
        return image, mask

def apply_augmentation(patches, label_patches):
    """
    Apply random augmentations (horizontal flip, vertical flip, etc.) to patches and labels.
    Augmented patches are added to the original dataset.

    Args:
        patches (list of torch.Tensor): Original image patches.
        label_patches (list of torch.Tensor): Original label patches.

    Returns:
        combined_patches (list of torch.Tensor): Original + augmented image patches.
        combined_labels (list of torch.Tensor): Original + augmented label patches.
    """
    augmented_patches = []
    augmented_labels = []

    for patch, label in zip(patches, label_patches):
        # Convert to tensors for augmentation
        image_tensor = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        label_tensor = torch.tensor(label, dtype=torch.long)  # (H, W)

        # Append original patch and label
        augmented_patches.append(image_tensor)
        augmented_labels.append(label_tensor)

        # Randomly apply horizontal flip
        if random.random() > 0.5:
            h_flip_image = torch.flip(image_tensor, dims=[2])  # Flip width
            h_flip_label = torch.flip(label_tensor, dims=[1])
            augmented_patches.append(h_flip_image)
            augmented_labels.append(h_flip_label)

        # Randomly apply vertical flip
        if random.random() > 0.5:
            v_flip_image = torch.flip(image_tensor, dims=[1])  # Flip height
            v_flip_label = torch.flip(label_tensor, dims=[0])
            augmented_patches.append(v_flip_image)
            augmented_labels.append(v_flip_label)

        # Randomly apply both flips
        if random.random() > 0.5:
            hv_flip_image = torch.flip(image_tensor, dims=[1, 2])  # Flip both height and width
            hv_flip_label = torch.flip(label_tensor, dims=[0, 1])
            augmented_patches.append(hv_flip_image)
            augmented_labels.append(hv_flip_label)

    # Combine original and augmented data
    combined_patches = augmented_patches
    combined_labels = augmented_labels

    return combined_patches, combined_labels


def create_dataset(patches, label_patches, num_classes, augment=True, min_classes=2):
    """
    Create dataset with optional filtering and augmentation.

    Args:
        patches (list of torch.Tensor): Image patches.
        label_patches (list of torch.Tensor): Label patches.
        num_classes (int): The number of classes for segmentation.
        augment (bool): Whether to augment the data (default: True).
        min_classes (int): Minimum number of classes required in a patch.

    Returns:
        dataset (Dataset): PyTorch Dataset containing image-label pairs.
    """
    # Filter patches based on the required number of classes
    patches, label_patches = filter_patches(patches, label_patches, min_classes)
    # Augment data if enabled
    if augment:
        patches, label_patches = apply_augmentation(patches, label_patches)
    return PatchSegmentationDataset(patches=patches, label_patches=label_patches, num_classes=num_classes)


def predict_on_patch(model, patch, device):
    """
    Predict on a single patch using the pre-trained binary segmentation model.
    Converts the patch to the correct input format, normalizes it, and performs prediction on the specified device.
    """
    # Ensure the patch is in the right format: (C, H, W)
    patch = np.transpose(patch, (2, 0, 1))  # Convert to (C, H, W)
    
    # Normalize the patch by dividing by 5000
    patch = patch / 50000.0
    
    # Convert the patch to a tensor and add batch dimension (1, C, H, W)
    patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(patch_tensor)  # Model inference
        probabilities = torch.sigmoid(output).squeeze(0).cpu()  # Apply sigmoid to get probabilities
        predicted_mask = (probabilities > 0.5).float()  # Threshold probabilities to obtain binary mask
    
    return predicted_mask.numpy()  # Convert prediction back to numpy array

def process_image(image_path_rgb, output_path, model, patch_size=256, device='cpu'):
    """
    Predict on a large image using non-overlapping patches and stitch back together.
    """
    with rasterio.open(image_path_rgb) as src_rgb:
        # Get image dimensions and metadata
        width, height = src_rgb.width, src_rgb.height
        profile = src_rgb.profile

        # Calculate the padding needed
        pad_height = (patch_size - height % patch_size) % patch_size
        pad_width = (patch_size - width % patch_size) % patch_size

        # Read the entire image and add padding if necessary
        image_rgb = src_rgb.read([1, 2, 3, 4,5,6,7])  # Read all bands
        image_rgb = np.moveaxis(image_rgb, 0, -1)  # Convert to (H, W, C)

        # Pad the image to make it a multiple of patch_size
        padded_image = np.pad(image_rgb, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

        # Get padded image dimensions
        padded_height, padded_width, _ = padded_image.shape

        # Initialize the prediction array (same size as the padded image)
        prediction_image = np.zeros((padded_height, padded_width), dtype=np.float32)

        # Iterate over patches
        for top in range(0, padded_height, patch_size):
            for left in range(0, padded_width, patch_size):
                # Define the patch boundaries
                bottom = min(top + patch_size, padded_height)
                right = min(left + patch_size, padded_width)
                
                # Extract the patch
                patch_rgb = padded_image[top:bottom, left:right]
                
                # Predict on the patch
                prediction = predict_on_patch(model, patch_rgb, device)
                
                # Store the prediction in the output array
                prediction_image[top:bottom, left:right] = prediction

        # Trim the padding from the prediction image
        prediction_image = prediction_image[:height, :width]

        # Write the prediction to a new GeoTIFF file
        profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction_image, 1)
