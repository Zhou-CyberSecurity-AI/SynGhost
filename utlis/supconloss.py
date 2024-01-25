import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True, device='cuda:0'):
        super(SupConLoss, self).__init__()

        self.temperature = temperature
        self.device = device
        self.scale_by_temperature = scale_by_temperature
        
    def forward(self, features, labels=None, mask=None):
        """_summary_

        Args:
            features (_type_): feature-> size=(batch_size, hidden_dim)
            labels (_type_, optional): ground_truth label
            mask (_type_, optional): mask for contrastive learning, size=(batch_size, batch_size)
        """
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        
        """_summary_

        label and mask:
            case1: Labels and mask cannot define values at the same time, because if there is a label, then the mask needs to be obtained according to the Label
            case2: If there are no labels and no mask, it is unsupervised learning. The mask is a matrix with a diagonal of 1, indicating that (i,i) belong to the same class
            case3: If labels are given, the mask is obtained according to the label. When the labels of the two samples i and j are equal, mask_{i,j}=1
        """
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)
        
        # compute logits that Dot product similarity between pairwise samples
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # anchor_dot_contrast - max_value of each line
        logits = anchor_dot_contrast - logits_max.detach() 
        exp_logits = torch.exp(logits)
        
        # construct mask for (i, 
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(self.device)
        # remove itself mask
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        
        num_positives_per_row  = torch.sum(positives_mask , axis=1)
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)  
        # denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True)
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        
        # compute positive sample average log-likehood
        log_probs = torch.sum(log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
        