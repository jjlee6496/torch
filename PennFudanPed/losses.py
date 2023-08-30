import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)