import torch
import argparse
import json
import torch.backends.cudnn as cudnn
import numpy as np
import scipy.stats as stats

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import ArchPerfDataset
from network import AutoEncoder

parser = argparse.ArgumentParser(description='PyTorch Estimator Evaluation')
parser.add_argument('--data_path', type=str, default='./data/', help='dataset path')
parser.add_argument('--train_ratio', type=float, default=0.9, help='ratio of train data')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--num_workers', type=int, default='4')
parser.add_argument('--save_name', type=str, default='exp1')
parser.add_argument('--encode_dimension', type=int, default=11)
parser.add_argument('--dropout_ratio', type=float, default=0.5)
parser.add_argument('--model_pth', type=str, default='cplfw_rank_epoch_inteval5_model.pth')
parser.add_argument('--model_path_index', type=int, default=-1)

args = parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    cudnn.deterministic = True 

@torch.no_grad()
def val_epoch(model, val_loader, epoch=0):
    running_ktau = 0
    for step, (archs, targets) in enumerate(val_loader):
        
        archs, targets = archs.cuda(), targets.cuda()

        outputs = model(archs)
        outputs = outputs.squeeze(1)

        ktau, p_value = stats.kendalltau(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        running_ktau += ktau

    epoch_ktau = running_ktau/(step+1)
    print('[Validate] Epoch {}: ktau: {:.4f}'.format(epoch+1, epoch_ktau))

@torch.no_grad()
def test(model, test_loader):
    
    total_output = []
    for step, archs in enumerate(test_loader):

        archs = archs.cuda()

        outputs = model(archs)
        outputs = list(outputs.squeeze(1).detach().cpu().numpy())
        total_output = total_output + outputs

    return total_output

def main(target_type):
    
    torch.cuda.set_device(args.gpu)
    set_seed(args.seed)
    
    g_cpu = torch.Generator()
    g_cpu.manual_seed(args.seed)

    print('before dataloader init',torch.randn(2,3))

    train_data = ArchPerfDataset(root=args.data_path, target_type=target_type, train=True, encode_dimension=args.encode_dimension)
    test_data = ArchPerfDataset(root=args.data_path, target_type=target_type, train=False, encode_dimension=args.encode_dimension)

    indices = list(range(len(train_data)))
    split = int(np.floor(args.train_ratio*len(train_data)))
    
    val_loader = DataLoader(train_data, 
                            batch_size=args.batch_size,
                            sampler=SubsetRandomSampler(indices[split:], generator=g_cpu),
                            pin_memory=True,
                            num_workers=args.num_workers,
                            drop_last=True
                            )

    test_loader = DataLoader(test_data, 
                            batch_size=args.batch_size,
                            pin_memory=True,
                            shuffle=False,
                            num_workers=args.num_workers,
                            drop_last=False
                            )
    print('after dataloader init',torch.randn(2,3))
    
    all_model_weights = torch.load('./results/{}'.format(args.model_pth))
    all_model_index = sorted(all_model_weights.keys())
    
    model_key = all_model_index[args.model_path_index]
    print(all_model_index)

    model = AutoEncoder(dropout=args.dropout_ratio).cuda()
    
    print("using model with val acc {}".format(model_key))
    model.load_state_dict(all_model_weights[model_key])

    model.eval()
    val_epoch(model, val_loader)
    model.eval()
    total_output = test(model, test_loader)

    return total_output

def norm_list(scores):
    scores_ls_sort=scores.tolist()
    scores_ls=scores.tolist()
    scores_ls_sort.sort()
    rank_number=[]
    for item in scores_ls:
        rank=scores_ls_sort.index(item)
        rank_number.append(rank)
    return rank_number

if __name__ == '__main__':

    task_list = ["cplfw_rank","vehicleid_rank","dukemtmc_rank","market1501_rank","msmt17_rank","veri_rank","veriwild_rank","sop_rank"]
    
    with open('./data/CVPR_2022_NAS_Track2_test.json', 'r') as f:
        test_data = json.load(f)
    
    for data_type in task_list:

        if data_type == 'cplfw_rank':
            args.batch_size = 8
            args.train_ratio = 0.7
            args.seed = 4
            args.dropout_ratio = 0.4
            args.model_pth = 'cplfw_final.pth'
            args.model_path_index = -1
        elif data_type == 'vehicleid_rank':
            args.batch_size = 25
            args.train_ratio = 0.9
            args.seed = 4
            args.dropout_ratio = 0.4
            args.model_pth = 'vehicleid_final.pth'
            args.model_path_index = -2
        elif data_type == 'dukemtmc_rank':
            args.batch_size = 25
            args.train_ratio = 0.9
            args.seed = 4
            args.dropout_ratio = 0.4
            args.model_pth = 'dukemtmc_final.pth'
            args.model_path_index = -3
        elif data_type == 'market1501_rank':
            args.batch_size = 32
            args.train_ratio = 0.9
            args.seed=1
            args.dropout_ratio=0.5
            args.model_pth = 'market1501_final.pth'
            args.model_path_index = -7
        elif data_type == 'msmt17_rank':
            args.batch_size = 32
            args.train_ratio = 0.8
            args.seed=1
            args.dropout_ratio=0.4
            args.model_pth = 'msmt17_final.pth'
            args.model_path_index = -4
        elif data_type == 'veri_rank':
            args.batch_size = 32
            args.train_ratio = 0.8
            args.seed=0
            args.dropout_ratio=0.5
            args.model_pth = 'veri_final.pth'
            args.model_path_index = -1
        elif data_type == 'veriwild_rank':
            args.batch_size = 32
            args.train_ratio = 0.8
            args.seed=3
            args.dropout_ratio=0.5
            args.model_pth = 'veriwild_final.pth'
            args.model_path_index = -43
        elif data_type == 'sop_rank':
            args.batch_size = 50
            args.train_ratio = 0.8
            args.seed=4
            args.dropout_ratio=0.5
            args.model_pth = 'sop_final.pth'
            args.model_path_index = -4
        print(args)
        
        print('start to process task {}'.format(data_type))
        
        total_output = main(data_type)
        total_output = norm_list(np.array(total_output))

        for i, key in enumerate(test_data.keys()):
            test_data[key][data_type] = int(total_output[i])
        
    print('Ready to save results!')
    with open('./Track2_submitA_{}.json'.format(args.save_name), 'w') as f:
        json.dump(test_data, f)


