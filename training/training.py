#This is the training and validation loop function

import torch
import numpy as np 
from sklearn import metrics

## These functions were created partially by AI under careful monitoring. Comments are my own. 

def accuracy(target, pred):
    return metrics.accuracy_score(
        target.detach().cpu().numpy(),  #Detact to numpy
        pred.detach().cpu().numpy()     #Detact to numpy
    )

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

def train_model(n_epochs, model, loss_fn, optimizer, train_loader, test_loader, device, train_val_steps=20):

    ''' 
    This training function trains our ResUnet model by looping through our training data loader, processing each batch
    and calculating the loss and gradients. 
    Then it updates the parameters and computes accuracy metrics both from train and validation (called test here) patches.
    '''

    step = 0 
    
    model.train()
    train_acc = []
    val_acc = []
    train_iou = [] 
    val_iou = [] 

    valid_accuracy = 0 
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")

        train_acc_batches = []
        train_iou_batches = []
        train_losses = []

        for logits, targets in train_loader:
            logits, targets = logits.to(device), targets.to(device)

            targets = targets.unsqueeze(1)

            logits = model(logits)
            loss = loss_fn(logits, targets.float())
            
            optimizer.zero_grad()
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            step += 1

            predictions = torch.sigmoid(logits) > 0.5
            train_accuracy = accuracy(targets.view(-1), predictions.view(-1))
            train_acc_batches.append(train_accuracy)

            train_iou_score = compute_iou(logits, targets)
            train_iou_batches.append(train_iou_score)

            if step % train_val_steps == 0:

                valid_accuracies_batches = []
                valid_iou_batches = [] 
                val_losses = []  

                with torch.no_grad():

                    model.eval() 

                    for val_inputs, val_targets in test_loader:
                        val_inputs,  val_targets = val_inputs.to(device), val_targets.to(device)
                        val_targets = val_targets.unsqueeze(1)  

                        val_logits = model(val_inputs)

                        val_loss = loss_fn(val_logits, val_targets.float()) 
                        val_losses.append(val_loss.item()) 

                        val_predictions = torch.sigmoid(val_logits) > 0.5

                        valid_accuracies_batches.append(
                            accuracy(val_targets.view(-1), val_predictions.view(-1)) * len(val_inputs)
                        ) 
                        val_iou_score = compute_iou(val_logits, val_targets)
                        valid_iou_batches.append(val_iou_score)
                
                valid_accuracy = np.sum(valid_accuracies_batches) / len(test_loader.dataset)
                valid_iou_score = np.mean(valid_iou_batches)
                print(f"Step {step:<5} | Train Accuracy: {train_accuracy:.4f} | Train IoU: {np.mean(train_iou_batches):.4f} | "
                        f"Val Accuracy: {valid_accuracy:.4f} | Val IoU: {valid_iou_score:.4f} | Train Loss: {loss.item():.4f} | Val Loss: {np.mean(val_losses):.4f}")

        # Compute and store epoch-level accuracies
        epoch_train_accuracy = np.mean(train_acc_batches)
        epoch_train_iou = np.mean(train_iou_batches)
        train_acc.append(epoch_train_accuracy)
        train_iou.append(epoch_train_iou)

        val_acc.append(valid_accuracy)
        val_iou.append(valid_iou_score)
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch + 1:<5} - Epoch avg: | Train Accuracy: {epoch_train_accuracy:.4f} | Train IoU: {epoch_train_iou:.4f} | Val Accuracy: {valid_accuracy:.4f} | Val IoU: {valid_iou_score:.4f} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    print("Training complete")

    return train_acc, val_acc, train_iou, val_iou