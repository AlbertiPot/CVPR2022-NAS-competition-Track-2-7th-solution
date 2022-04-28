import torch
import argparse
import json
import torch.backends.cudnn as cudnn
import numpy as np
import scipy.stats as stats

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import ArchPerfDataset
from network import Predictor

parser = argparse.ArgumentParser(description='PyTorch Estimator Training')
parser.add_argument('--data_path', type=str, default='./data/', help='dataset path')
parser.add_argument('--train_ratio', type=float, default=0.9, help='ratio of train data')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--num_workers', type=int, default='4')
parser.add_argument('--save_name', type=str, default='exp1')

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

    train_data = ArchPerfDataset(root=args.data_path, target_type=target_type, train=True)
    test_data = ArchPerfDataset(root=args.data_path, target_type=target_type, train=False)

    indices = list(range(len(train_data)))
    split = int(np.floor(args.train_ratio*len(train_data)))
    
    val_loader = DataLoader(train_data, 
                            batch_size=args.batch_size,
                            sampler=SubsetRandomSampler(indices[split:]),
                            pin_memory=True,
                            num_workers=args.num_workers,
                            drop_last=True
                            )

    test_loader = DataLoader(test_data, 
                                batch_size=args.batch_size,
                                pin_memory=True,
                                shuffle=False,
                                num_workers=args.num_workers
                                )

    model = Predictor().cuda()
    model.load_state_dict(torch.load('./results/{}_seed{}_{}.pth'.format(target_type, args.seed, args.save_name)))

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
        rank=scores_ls_sort.index(item)+1
        rank_number.append(rank)
    return rank_number

if __name__ == '__main__':
    
    task_list = ["cplfw_rank", 
                "market1501_rank", 
                "dukemtmc_rank",
                "msmt17_rank",
                "veri_rank",
                "vehicleid_rank",
                "veriwild_rank",
                "sop_rank"]
    # task_list = ["cplfw_rank"]
    
    with open('./data/CVPR_2022_NAS_Track2_test.json', 'r') as f:
        test_data = json.load(f)
    
    for data_type in task_list:
        print('start to process task {}'.format(data_type))
        
        total_output = main(data_type)
        total_output = norm_list(np.array(total_output))

        for i, key in enumerate(test_data.keys()):
            test_data[key][data_type] = int(total_output[i])
        
    print('Ready to save results!')
    with open('./eval_Track2_submitA_seed{}_{}.json'.format(args.seed,args.save_name), 'w') as f:
        json.dump(test_data, f)


