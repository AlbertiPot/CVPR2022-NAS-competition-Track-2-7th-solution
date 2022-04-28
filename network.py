import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, input_c=25, h=1, w=3, dropout=0.5):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_c*h*w, 64),
            
            nn.Linear(64,48),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(48,32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32,16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(16,1)
            )

    def forward(self, x):
        out = self.net(x)
        return out