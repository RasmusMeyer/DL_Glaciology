import torch
import torch.nn as nn


## This python script contains loss functions
## These are used to evaluate the models performance

## For binary semantic image segmentation, we create test two custom loss functions and BCE: 

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super().__init__()
        """
        When to use: helps with hard-to-classify pixels, by giving them extra weight. 
        logits: raw outputs from NN before applying activation sigmoid function.
        targets: this is the 0 or 1 true binary values from the labels.
        Alpha: weighing applied to each loss, to take inbalanced classes into consideration. 
        Gamma: adjust how much the model should focus on hard-to-classify objects
        """
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')

        # we apply the focal factor to the bce_loss to skew it towards the hard-to-classify pixels
        f1 = torch.where(targets == 1, probs, 1 - probs)
        focal_factor = (1 - f1) ** self.gamma

        #calculate new loss
        loss = self.alpha * focal_factor * bce

        return loss.mean()
    
class BinaryDiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        """
        When to use: takes the whole patch into account by calculating the overlap of the classes for each batch. 

        logits: raw outputs from NN before applying activation sigmoid function
        targets: this is the 0 or 1 true binary values from the labels
        smooth: we smooth to prevent division by 0
        """
        self.smooth = smooth
    def forward(self, logits, targets):
            probs = torch.sigmoid(logits)
            probs = probs.view(-1)
            targets = targets.view(-1)

            #Calculate intersection
            intersection = torch.sum(probs * targets)
            
            #Dice coefficient, which is the overlap between probs and targets
            dice_coef = (2 * intersection + self.smooth) / (torch.sum(probs) + torch.sum(targets) + self.smooth)

            return 1 - dice_coef