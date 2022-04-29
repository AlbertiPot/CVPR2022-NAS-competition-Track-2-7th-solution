import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter 不是参数
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid)) # buffer:存储模型的状态state而不是需要更新梯度的参数，这里的位置编码是state而非模型的参数

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] # 每个位置position,对应一个d_hid=512维的向量

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])   # (200,512) 共200个位置，每个位置一个512长度的矢量
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)                                       # (1,200,512)

    def forward(self, x):   # x =[16, 25, 1, 3]  → x = [bsz, c, d_hidd]
        x = x.squeeze(2)
        return x + self.pos_table[:, :x.size(1)].clone().detach()                                   # 将pos_table中存储的位置参数复制一份，与输入相加。detach方法将位置编码从当前计算图中剥离，因此将不会跟踪梯度

class AutoEncoder(nn.Module):
    def __init__(self, input_c=25, h=1, w=3, dropout=0.5):
        super(AutoEncoder, self).__init__()
        self.pos = PositionalEncoding(d_hid=h*w, n_position=30)
        self.fc1 = nn.Linear(input_c*h*w, 64)
        
        self.net = nn.Sequential(            
            nn.Linear(64,32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32,16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(16,32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            )
        self.fc2 = nn.Linear(32,1)

    def forward(self, x):   # x =[bsz, c, h,w]
        out = self.pos(x)   # x =[bsz, c, d_hidden]
        out = out.view(out.size(0),-1)  # x =[bsz, c*h*w]
        out = self.fc1(out)
        out = self.net(out)
        out = self.fc2(out)
        return out

class Encoder(nn.Module):
    def __init__(self, input_c=25, h=1, w=3, dropout=0.5):
        super(Encoder, self).__init__()
        self.pos = PositionalEncoding(d_hid=h*w, n_position=30)
        self.fc1 = nn.Linear(input_c*h*w, 64)
        
        self.net = nn.Sequential(            
            nn.Linear(64,48),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(48,32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32,16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            )
        self.fc2 = nn.Linear(16,1)

    def forward(self, x):   # x =[bsz, c, h,w]
        out = self.pos(x)   # x =[bsz, c, d_hidden]
        out = out.view(out.size(0),-1)  # x =[bsz, c*h*w]
        out = self.fc1(out)
        out = self.net(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    encode = PositionalEncoding(d_hid=3, n_position=30)
    a = torch.randn(16,25,1,3).squeeze(2)
    b= encode(a)