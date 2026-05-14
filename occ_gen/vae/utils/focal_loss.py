import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze()
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-10)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

if __name__ == "__main__":
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    focal_loss = FocalLoss(alpha=1, gamma=2)
    loss = focal_loss(logits, targets)
    print(f"Focal Loss: {loss.item()}")