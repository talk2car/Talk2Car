import torch
import torch.nn as nn

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, scores, gt):
        # Input:
        #   scores: B x R   (Similarity score for every region)
        #   gt: B x R       (IoU > 0.5 for every sample)
        # Output:
        #   loss            (Loss value)
        #   pred            (Prediction = Region with highest score)
        
        pred = torch.argmax(scores, dim = 1)
        score = scores.view(-1)
        gt = gt.view(-1)
    
        pos_mask, neg_mask = (gt == 1), (gt == 0)

        loss = self.loss(score, gt)
        pos_loss = torch.masked_select(loss, pos_mask)
        neg_loss = torch.masked_select(loss, neg_mask)
        total_loss = pos_loss.mean() + neg_loss.mean() # Avoid bias

        return total_loss, pred
