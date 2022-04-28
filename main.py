import torch
import argparse
import copy
import json
import torch.nn as nn
import torch.optim as optim
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
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=3e-4)
parser.add_argument('--num_epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--log_interval', type=int, default=100)
# parser.add_argument('--target_type', type=str, default='market1501_rank', help='8 target missions')
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

# Define Loss
def pair_loss(outputs, labels): # output.shape = torch.Size([1023]) labels = torch.Size([1023])
    
    output = outputs.unsqueeze(1)
    output1 = output.repeat(1,outputs.shape[0])
    label = labels.unsqueeze(1)
    label1 = label.repeat(1,labels.shape[0])
    tmp = (output1-output1.t())*torch.sign(label1-label1.t())
    tmp = torch.log(1+torch.exp(-tmp))
    eye_tmp = tmp*torch.eye(len(tmp)).cuda()
    new_tmp = tmp - eye_tmp
    loss = torch.sum(new_tmp)/(outputs.shape[0]*(outputs.shape[0]-1))
    return loss

def train_epoch(model, criterion, optimizer, train_loader, epoch, log_interval):
    
    running_loss = 0
    running_ktau = 0
    for step, (archs, targets) in enumerate(train_loader):
        
        archs, targets = archs.cuda(), targets.cuda()
        optimizer.zero_grad()
        
        outputs = model(archs)
        outputs = outputs.squeeze(1)

        # outputs.shape=torch.Size([16]) targets.shape = torch.Size([16])
        # loss = criterion(outputs, targets)
        loss = pair_loss(outputs, targets) 

        loss.backward()
        optimizer.step()

        ktau, p_value = stats.kendalltau(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        running_ktau += ktau
        running_loss += loss.item()*archs.size(0)

    epoch_loss = running_loss/(len(train_loader)*archs.size(0))
    epoch_ktau = running_ktau/(step+1)

    if (epoch+1)%log_interval ==0 or epoch ==0:
        print('[Train] Epoch {}: Loss: {:.4f} ktau: {:.4f}'.format(epoch+1, epoch_loss, epoch_ktau))

@torch.no_grad()
def val_epoch(model, val_loader, epoch, log_interval):

    running_ktau = 0
    for step, (archs, targets) in enumerate(val_loader):
        
        archs, targets = archs.cuda(), targets.cuda()

        outputs = model(archs)
        outputs = outputs.squeeze(1)

        ktau, p_value = stats.kendalltau(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        running_ktau += ktau

    epoch_ktau = running_ktau/(step+1)
    if (epoch+1)%log_interval ==0 or epoch ==0:
        print('[Validate] Epoch {}: ktau: {:.4f}'.format(epoch+1, epoch_ktau))
        
    return epoch_ktau

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

    train_loader = DataLoader(train_data, 
                            batch_size=args.batch_size,
                            sampler=SubsetRandomSampler(indices[:split]),
                            pin_memory=True,
                            num_workers=args.num_workers,
                            drop_last=True
                            )
    
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
    
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_ktau = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    for eps in range(args.num_epochs):

        model.train()
        train_epoch(model, criterion, optimizer, train_loader, eps, args.log_interval)

        model.eval()
        epoch_ktau = val_epoch(model, val_loader, eps, args.log_interval)
        
        if epoch_ktau > best_ktau:
                best_ktau = epoch_ktau
                torch.save(model.state_dict(), './results/{}_seed{}_{}.pth'.format(target_type, args.seed, args.save_name))
                best_model_weights = copy.deepcopy(model.state_dict())

    print('Best train KTau: {:4f} on task {}'.format(best_ktau, target_type))

    model.load_state_dict(best_model_weights)

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
    with open('./train_Track2_submitA_seed{}_{}.json'.format(args.seed,args.save_name), 'w') as f:
        json.dump(test_data, f)
        