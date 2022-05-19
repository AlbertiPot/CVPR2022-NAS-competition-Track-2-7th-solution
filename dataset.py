import os
import json
import numpy as np
from torch.utils.data import Dataset


def convert_arch(arch_str, encode_dimension):
    arch_str = list(arch_str)

    onehot_as = [None for _ in range(0, 37)]

    for i in range(1, 37):
        if i % 3 == 1:
            onehot_as[i] = 2 + int(arch_str[i])
        if i % 3 == 2:
            onehot_as[i] = 5 + int(arch_str[i])
        if i % 3 == 0 and arch_str[i] == '1':
            onehot_as[i] = 9

    layer_num = arch_str[0]
    if layer_num == 'j':
        onehot_as[0] = 0
        onehot_as[-3 * 2:] = [10 for _ in range(0, 6)]
    elif layer_num == 'k':
        onehot_as[0] = 1
        onehot_as[-3 * 1:] = [10 for _ in range(0, 3)]
    elif layer_num == 'l':
        onehot_as[0] = 2

    decode_arch = np.eye(encode_dimension, 10)[onehot_as]
    decode_arch = np.array(decode_arch, dtype=np.float32)
    return (decode_arch)  # list (37, 10)


class ArchPerfDataset(Dataset):
    def __init__(self, root, target_type, train=True, encode_dimension=11):
        self.train = train
        self.encode_dimension = encode_dimension

        assert target_type in [
            "cplfw_rank",
            "market1501_rank",
            "dukemtmc_rank",
            "msmt17_rank",
            "veri_rank",
            "vehicleid_rank",
            "veriwild_rank",
            "sop_rank",
        ], 'Wrong target type'

        self.target_type = target_type

        if self.train:
            data_pth = os.path.join(root, 'CVPR_2022_NAS_Track2_train.json')
        else:
            data_pth = os.path.join(root, 'CVPR_2022_NAS_Track2_test.json')

        self._load_meta(data_pth)

        self.archs = []
        self.targets = []

        self._extract_data()

    def _load_meta(self, data_pth):
        print('load data from:', data_pth)

        with open(data_pth, 'r') as f:
            self.data = json.load(f)

    def _extract_data(self):
        for _, k in enumerate(self.data):
            self.archs.append(self.data[k]['arch'])
            if self.train:
                self.targets.append(self.data[k][self.target_type])

    def __getitem__(self, index: int):
        arch = convert_arch(self.archs[index], self.encode_dimension)

        if self.train:
            target = self.targets[index]
            return arch, np.array(target, dtype=np.float32)

        return arch

    def __len__(self):
        return len(self.data.keys())


if __name__ == '__main__':
    a = ArchPerfDataset('./data', 'sop_rank', True)
    a.__getitem__(1)
