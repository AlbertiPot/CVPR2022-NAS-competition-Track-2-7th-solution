import os
import json
import numpy as np
from torch.utils.data import Dataset

def convert_arch(arch):
        arch_str = list(arch)
        
        for i in range(0,12):
            del arch_str[-1-2*i]
        
        decode_arch = [[]]        
        for elm in arch_str:
            if elm == 'j' or elm =='1':
                decode_arch[0].append([1,0,0])
            elif elm == 'k' or elm =='2':
                decode_arch[0].append([0,1,0])
            elif elm == 'l' or elm =='3':
                decode_arch[0].append([0,0,1])
            elif elm=='0':
                decode_arch[0].append([0,0,0])
        
        decode_arch = np.array(decode_arch,dtype=np.float32).transpose(1,0,2)
        
        return decode_arch

class ArchPerfDataset(Dataset):
    def __init__(self, root, target_type, train=True):
        self.train = train

        assert target_type in ["cplfw_rank", 
                                "market1501_rank", 
                                "dukemtmc_rank",
                                "msmt17_rank",
                                "veri_rank",
                                "vehicleid_rank",
                                "veriwild_rank",
                                "sop_rank",], 'Wrong target type'
        
        self.target_type = target_type

        
        if self.train:
            data_pth = os.path.join(root,'CVPR_2022_NAS_Track2_train.json')
        else:
            data_pth = os.path.join(root,'CVPR_2022_NAS_Track2_test.json')
        
        self._load_meta(data_pth)

        self.archs = []
        self.targets = []
        
        self._extract_data()
    

    def _load_meta(self, data_pth):
        print('load data from:',data_pth)
        
        with open(data_pth,'r') as f:
            self.data = json.load(f)
    
    def _extract_data(self):
        for _, k in enumerate(self.data):
            self.archs.append(self.data[k]['arch'])
            if self.train:
                self.targets.append(self.data[k][self.target_type])
    
    def __getitem__(self, index: int):
        arch = convert_arch(self.archs[index])
        
        if self.train:
            target = self.targets[index]
            return arch, np.array(target,dtype=np.float32)
        
        return arch
    
    def __len__(self):
        return len(self.data.keys())

if __name__ == '__main__':
    a = ArchPerfDataset('./data','sop_rank',True)
    a.__getitem__(1)


