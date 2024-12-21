import torch
import rasterio
import numpy 

def compute_iou(predictions, targets, threshold=0.5):
    '''
    This function computes thet intersect over union validation metric by
    1) using sigmoid to get binary predictions instead of probabilities
    2) Calculate the IoU, which is the overlap between the prediction and targets compared to the total area they cover. 
    '''
    preds = torch.sigmoid(predictions) > threshold 
    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets) - intersection
    iou = intersection / union if union != 0 else torch.tensor(0.0)

    return iou.item()

def load_tif_as_tensor(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)
    return torch.tensor(data, dtype=torch.float32)

def calculate_iou_for_images(predictions_path, labels_path, threshold=0.5):
    preds_tensor = load_tif_as_tensor(predictions_path)
    labels_tensor = load_tif_as_tensor(labels_path)
    iou = compute_iou(preds_tensor, labels_tensor, threshold)
    return iou