import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import argparse
import copy
import utils.log
from PointDA.data.dataloader import ScanNet, ModelNet, ShapeNet, label_to_idx
from PointDA.Models import PointNet, DGCNN
from utils import pc_utils
from MLSP import PCM, mlsp
import math
import pcl
import time
import functools
import torch.nn.functional as F
NWORKERS=4
MAX_LOSS = 9 * (10**9)

def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--exp_name', type=str, default='MLSP',  help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--dataroot', type=str, default='./data', metavar='N', help='data path')
parser.add_argument('--src_dataset', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--epochs', type=int, default=150, help='number of episode to train')
parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Model to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--DefRec_dist', type=str, default='volume_based_voxels', metavar='N',
                    choices=['volume_based_voxels', 'volume_based_radius'],
                    help='distortion of points')
parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
parser.add_argument('--DefRec_on_src', type=str2bool, default=False, help='Using DefRec in source')

parser.add_argument('--apply_PCM', type=str2bool, default=True, help='Using mixup in source')
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--DefRec_weight', type=float, default=0.5, help='weight of the DefRec loss')
parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--DefRec_on_trgt', type=str2bool, default=False, help='Using DefRec in target')

parser.add_argument('--Norm_on_trgt', type=str2bool, default=False, help='Using normal prediction in source')
parser.add_argument('--normal_pred_weight', type=float, default=0.5, help='weight of the normal prediction loss')

parser.add_argument('--Scan_on_trgt', type=str2bool, default=False, help='Using normal prediction in source')
parser.add_argument('--Scan_Rec_weight', type=float, default=0.5, help='weight of the normal prediction loss')

parser.add_argument('--Density_on_trgt', type=str2bool, default=False, help='Using normal prediction in source')
parser.add_argument('--Density_weight', type=float, default=0.05, help='weight of the normal prediction loss')
parser.add_argument('--density_num_class', type=int, default=16, help='number of classes of density')
parser.add_argument('--pergroup', type=float, default=2, help='number of classes of density')
parser.add_argument('--radius', type=float, default=0.1, help='weight of the normal prediction loss')

parser.add_argument('--Density_normal_viainput', type=str2bool, default=False, help='Using normal prediction in source')
parser.add_argument('--Density_normal_viachamfer', type=str2bool, default=False, help='Using normal prediction in source')
parser.add_argument('--Density_normal_defpart', type=str2bool, default=False, help='Using normal prediction in source')
parser.add_argument('--Density_ondef', type=str2bool, default=False, help='Using normal prediction in source')
parser.add_argument('--Normal_ondef', type=str2bool, default=False, help='Using normal prediction in source')
parser.add_argument('--Density_normal_viainput_onsrc', type=str2bool, default=False, help='Using normal prediction in source')

parser.add_argument('--apply_SPL', type=str2bool, default=False, help='Using self-paced learning')
parser.add_argument('--gamma', type=float, default=0.1, help='threshold for pseudo label')
parser.add_argument('--apply_SPL_v2', type=str2bool, default=False, help='Using self-paced learning')
parser.add_argument('--gamma_v2', type=float, default=1.6366, help='threshold for pseudo label')

parser.add_argument('--num_class', type=float, default=10, help='number of classes of density')
parser.add_argument('--near', type=float, default=20, help='number of classes of density')
args = parser.parse_args()
args.encoder_type = None

# if args.Density_on_trgt:
if args.trgt_dataset=='shapenet':
    args.density_num_class=16
    args.radius=0.12
elif args.trgt_dataset=='modelnet':
    args.density_num_class=16
    args.radius=0.13
elif args.trgt_dataset=='scannet':
    args.density_num_class=16
    args.radius=0.135



#if args.Density_normal_viachamfer or args.Density_normal_viainput:
#    args.DefRec_on_trgt=False
# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

random.seed(1)
np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')

# ==================
# Calculate normal
# ==================
# if args.Norm_on_trgt:
#     cloud = pcl.PointCloud()

# 日志耗时装饰器
def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print('[%s] took %.2f s' % (func.__name__, (end - start)))
        return res

    return wrapper


# @log_execution_time
def radiusSearchNormalEstimation(cloud):
    ne = cloud.make_NormalEstimation()
    ne.set_RadiusSearch(0.1)
    normals = ne.compute()
    # print(normals.size, type(normals), normals[0], type(normals[0]))
    count = 0
    for i in range(0, normals.size):
        if (str(normals[i][0]) == 'nan'):
            continue
        count = count + 1
    # print(count)
    normals = normals.to_array()
    return normals

# @log_execution_time
def kSearchNormalEstimation(cloud,near = 20):
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_KSearch(near)
    normals = ne.compute()
    # print(normals.size, type(normals), normals[0])
    count = 0
    for i in range(0, normals.size):
        if (str(normals[i][0]) == 'nan'):
            continue
        count = count + 1
    normals = normals.to_array()
    # print(np.linalg.norm(normals,axis=1))
    # print(np.linalg.norm(normals[:,:3],axis=1))
    return normals[:,:3]

# ==================
# Read Data
# ==================
def split_set(dataset, domain, set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler

src_dataset = args.src_dataset
trgt_dataset = args.trgt_dataset
data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}

src_trainset = data_func[src_dataset](io, args.dataroot, 'train')
trgt_trainset = data_func[trgt_dataset](io, args.dataroot, 'train')
trgt_testset = data_func[trgt_dataset](io, args.dataroot, 'test')

# Creating data indices for training and validation splits:
src_train_sampler, src_valid_sampler = split_set(src_trainset, src_dataset, "source")
trgt_train_sampler, trgt_valid_sampler = split_set(trgt_trainset, trgt_dataset, "target")

# dataloaders for source and target
src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                               sampler=src_train_sampler, drop_last=True)
src_val_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
                             sampler=src_valid_sampler)
trgt_train_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                                sampler=trgt_train_sampler, drop_last=True)
trgt_val_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
                                  sampler=trgt_valid_sampler)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)

# ==================
# Init Model
# ==================
if args.model == 'pointnet':
    model = PointNet(args)
elif args.model == 'dgcnn':
    model = DGCNN(args)
else:
    raise Exception("Not implemented")

model = model.to(device)

# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
best_model = copy.deepcopy(model)

# ==================
# Optimizer
# ==================
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd) if args.optimizer == "SGD" \
    else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = CosineAnnealingLR(opt, args.epochs)
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch
# lookup table of regions means
lookup = torch.Tensor(pc_utils.region_mean(args.num_regions)).to(device)

def generate_trgt_pseudo_label(trgt_data, logits, threshold):
    batch_size = trgt_data.size(0)
    pseudo_label = torch.zeros(batch_size, 10).long()  # one-hot label
    sfm = nn.Softmax(dim=1)
    cls_conf = sfm(logits['cls'])
    mask = torch.max(cls_conf, 1)  # 2 * b
    for i in range(batch_size):
        index = mask[1][i]
        if mask[0][i] > threshold:
            pseudo_label[i][index] = 1

    return pseudo_label

def generate_trgt_pseudo_label_v2(trgt_data, logits, threshold):
    #[2.3026, 1.4612, 1.6365, 1.6280] #[1.6084, 1.6025]
    batch_size = trgt_data.size(0)
    pseudo_label = torch.zeros(batch_size, 10).long()  # one-hot label
    sfm = nn.Softmax(dim=1)
    cls_conf = sfm(logits['cls'])
    mask = torch.max(cls_conf, 1)
    entropys = -torch.sum(cls_conf*torch.log_softmax(cls_conf,dim=1),dim=1)

    # mask = torch.max(cls_conf, 1)  # 2 * b
    for i in range(batch_size):
        entropy = entropys[i]
        index = mask[1][i]
        if entropy<threshold:
            pseudo_label[i][index] = 1
    return pseudo_label

# ==================
# Validation/test
# ==================
def test(test_loader, model=None, set_type="Target", partition="Val", epoch=0):

    # Run on cpu or gpu
    count = 0.0
    print_losses = {'cls': 0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()
        test_pred = []
        test_true = []
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            logits = model(data, activate_DefRec=False)
            loss = criterion(logits["cls"], labels)
            print_losses['cls'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits["cls"].max(dim=1)[1]
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            count += batch_size
            batch_idx += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    test_acc = io.print_progress(set_type, partition, epoch, print_losses, test_true, test_pred)
    conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(label_to_idx.values())).astype(int)

    return test_acc, print_losses['cls'], conf_mat


# ==================
# Train
# ==================
src_best_val_acc = trgt_best_val_acc = best_val_epoch = 0
src_best_val_loss = trgt_best_val_loss = MAX_LOSS
best_model = io.save_model(model)

for epoch in range(args.epochs):
    model.train()

    # init data structures for saving epoch stats
    cls_type = 'mixup' if args.apply_PCM else 'cls'
    src_print_losses = {"total": 0.0, cls_type: 0.0}
    if args.DefRec_on_src:
        src_print_losses['DefRec'] = 0.0
    if args.Density_normal_viainput_onsrc:
        src_print_losses['def_normal_loss'] = 0.0
        src_print_losses['def_density_cls_loss'] = 0.0
        src_print_losses['def_density_mse_loss'] = 0.0
        src_print_losses['DefRec'] = 0.0
    trgt_print_losses = {'DefRec': 0.0}
    if args.Norm_on_trgt:
        trgt_print_losses['Normal'] = 0.0
    if args.Scan_on_trgt:
        trgt_print_losses['Rec_scan'] = 0.0
    if args.Density_on_trgt:
        trgt_print_losses['Density_cls'] = 0.0
        trgt_print_losses['Density_mse'] = 0.0
    if args.Density_normal_viachamfer or args.Density_normal_viainput:
        trgt_print_losses['def_normal_loss'] = 0.0
        trgt_print_losses['def_density_cls_loss'] = 0.0
        trgt_print_losses['def_density_mse_loss'] = 0.0
        criterion1 = nn.CrossEntropyLoss(reduction='none')
    if args.apply_SPL or args.apply_SPL_v2:
        trgt_print_losses['SPL'] = 0.0
        trgt_print_losses['selected_percent'] = 0.0

    src_count = trgt_count = 0.0

    batch_idx = 1
    for data1, data2 in zip(src_train_loader, trgt_train_loader):
        opt.zero_grad()

        #### source data ####
        if data1 is not None:
            src_data, src_label = data1[0].to(device), data1[1].to(device).squeeze()
            # change to [batch_size, num_coordinates, num_points]
            src_data = src_data.permute(0, 2, 1)
            batch_size = src_data.size()[0]
            src_data_orig = src_data.clone()
            device = torch.device("cuda:" + str(src_data.get_device()) if args.cuda else "cpu")

            if args.DefRec_on_src:
                src_data, src_mask = mlsp.deform_input(src_data, lookup, args.DefRec_dist, device)
                src_logits = model(src_data, activate_DefRec=True)
                loss = mlsp.calc_loss(args, src_logits, src_data_orig, src_mask)
                src_print_losses['DefRec'] += loss.item() * batch_size
                src_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            if args.apply_PCM:
                src_data = src_data_orig.clone()
                src_data, mixup_vals = PCM.mix_shapes(args, src_data, src_label)
                src_cls_logits = model(src_data, activate_DefRec=False)
                loss = PCM.calc_loss(args, src_cls_logits, mixup_vals, criterion)
                src_print_losses['mixup'] += loss.item() * batch_size
                src_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            else:
                src_data = src_data_orig.clone()
                # predict with undistorted shape
                src_cls_logits = model(src_data, activate_DefRec=False)
                loss = (1 - args.DefRec_weight) * criterion(src_cls_logits["cls"], src_label)
                src_print_losses['cls'] += loss.item() * batch_size
                src_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            if args.Density_normal_viainput_onsrc:
                src_data, src_label = data1[0].to(device), data1[1].to(device).squeeze()                
                normal_gt = []
                for i in range(src_data.size()[0]):
                    cloud = pcl.PointCloud()
                    pts = src_data[i].cpu().numpy()
                    cloud.from_array(np.array(pts, dtype=np.float32))
                    knn_norm = kSearchNormalEstimation(cloud, args.near)
                    normal_gt.append(knn_norm)
                normal_gt = torch.tensor(np.array(normal_gt),dtype=torch.float32).to(device) #[B,N,3]

                density_label, density_mse_label = mlsp.cal_density(src_data,radius=args.radius, num_cls=args.density_num_class, pergroup=args.pergroup)
                density_label = torch.tensor(density_label,dtype=torch.float).to(device)
                density_label = density_label.reshape(-1,args.density_num_class)
                density_mse_label = torch.tensor(density_mse_label,dtype=torch.float).to(device).reshape(-1)

                src_data = src_data.permute(0, 2, 1)
                src_data_orig = src_data.clone()
                device = torch.device("cuda:" + str(src_data.get_device()) if args.cuda else "cpu")
                src_data, src_mask = mlsp.deform_input(src_data, lookup, args.DefRec_dist, device)
                logits_pred = model(src_data, activate_density_normal_ondef=True)

                loss = mlsp.calc_loss(args, logits_pred, src_data_orig, src_mask)
                src_print_losses['DefRec'] += loss.item() * batch_size
                src_mask = src_mask.permute(0,2,1)
                if args.Density_normal_defpart:
                    mask_cord = src_mask[:, :, 0]
                else:
                    mask_cord = src_mask[:, :, 0]*26+1
                if args.Normal_ondef:
                    normal_pred = logits_pred['Normal']
                    normal_pred = F.normalize(normal_pred,p=2,dim=-1)
                    normal_gt = F.normalize(normal_gt,p=2,dim=-1)
                    norm_loss = -torch.sum(torch.abs(torch.sum(normal_pred*normal_gt,dim=-1))*mask_cord)/(torch.sum(mask_cord))
                    norm_loss = args.normal_pred_weight*norm_loss
                    src_print_losses['def_normal_loss'] += norm_loss.item() * batch_size
                    loss = loss+norm_loss
                if args.Density_ondef:
                    mask_cord = mask_cord.reshape(-1)
                    density_cls_loss, density_mse_loss = mlsp.densityloss(args,logits_pred,density_mse_label,density_label,mask=mask_cord)
                    # density_loss = args.Density_weight * torch.sum(criterion1(logits_pred["density"], density_gt)*mask_cord)/(batch_size*torch.sum(mask_cord))
                    src_print_losses['def_density_cls_loss'] += density_cls_loss.item() * batch_size
                    src_print_losses['def_density_mse_loss'] += density_mse_loss.item() * batch_size
                    loss = loss+density_mse_loss+density_cls_loss
                loss.backward()
            src_count += batch_size

        #### target data ####
        if data2 is not None:
            if args.DefRec_on_trgt:
                trgt_data, trgt_label = data2[0].to(device), data2[1].to(device).squeeze()
                trgt_data = trgt_data.permute(0, 2, 1)
                batch_size = trgt_data.size()[0]
                trgt_data_orig = trgt_data.clone()
                device = torch.device("cuda:" + str(trgt_data.get_device()) if args.cuda else "cpu")

                trgt_data, trgt_mask = mlsp.deform_input(trgt_data, lookup, args.DefRec_dist, device)
                trgt_logits = model(trgt_data, activate_DefRec=True)
                loss = mlsp.calc_loss(args, trgt_logits, trgt_data_orig, trgt_mask)
                trgt_print_losses['DefRec'] += loss.item() * batch_size
                loss.backward()


            if args.Norm_on_trgt:
                trgt_data, trgt_label = data2[0].to(device), data2[1].to(device).squeeze()
                normal_gt = []
                for i in range(trgt_data.size()[0]):
                    cloud = pcl.PointCloud()
                    pts = trgt_data[i].cpu().numpy()
                    cloud.from_array(np.array(pts, dtype=np.float32))
                    knn_norm = kSearchNormalEstimation(cloud)
                    normal_gt.append(knn_norm)
                normal_gt = torch.tensor(np.array(normal_gt),dtype=torch.float32).to(device)
                trgt_data = trgt_data.permute(0, 2, 1)
                trgt_logits = model(trgt_data, activate_normal=True)
                normal_pred = trgt_logits['Normal']
                loss = mlsp.calc_normal_loss(args, normal_pred, normal_gt)
                trgt_print_losses['Normal'] += loss.item() * batch_size
                loss.backward()

            if args.Scan_on_trgt:
                trgt_data, trgt_label = data2[0].to(device), data2[1].to(device).squeeze()
                trgt_data_orig = trgt_data.permute(0, 2, 1).clone()

                device = torch.device("cuda:" + str(trgt_data.get_device()) if args.cuda else "cpu")
                trgt_data, trgt_mask = mlsp.scan_input(trgt_data, device)
                trgt_data = trgt_data.permute(0, 2, 1)
                trgt_mask = trgt_mask.permute(0, 2, 1)
                trgt_logits = model(trgt_data, activate_scan=True)

                loss = mlsp.calc_scan_loss(args, trgt_logits, trgt_data_orig, trgt_mask)
                trgt_print_losses['Rec_scan'] += loss.item() * batch_size
                loss.backward()

            if args.Density_on_trgt:
                trgt_data, trgt_label = data2[0].to(device), data2[1].to(device).squeeze()
                trgt_data_orig = trgt_data.permute(0, 2, 1).clone()

                density_label, density_mse_label = mlsp.cal_density(trgt_data,radius=args.radius, num_cls=args.density_num_class, pergroup=args.pergroup)
                density_label = torch.tensor(density_label).to(device)#.long()
                density_label = density_label.reshape(-1,args.density_num_class)
                density_mse_label = torch.tensor(density_mse_label,dtype=torch.float).to(device).reshape(-1)

                trgt_logits = model(trgt_data_orig, activate_density=True)
                density_cls_loss, density_mse_loss = mlsp.densityloss(args,trgt_logits,density_mse_label,density_label)
                trgt_print_losses['Density_cls'] += density_cls_loss.item() * batch_size
                trgt_print_losses['Density_mse'] += density_mse_loss.item() * batch_size
                loss = density_cls_loss + density_mse_loss
                loss.backward()

            if args.Density_normal_viainput:
                trgt_data, trgt_label = data2[0].to(device), data2[1].to(device).squeeze()
                normal_gt = []
                for i in range(trgt_data.size()[0]):
                    cloud = pcl.PointCloud()
                    pts = trgt_data[i].cpu().numpy()
                    cloud.from_array(np.array(pts, dtype=np.float32))
                    knn_norm = kSearchNormalEstimation(cloud, args.near)
                    normal_gt.append(knn_norm)
                normal_gt = torch.tensor(np.array(normal_gt),dtype=torch.float32).to(device) #[B,N,3]

                density_label, density_mse_label = mlsp.cal_density(trgt_data,radius=args.radius, num_cls=args.density_num_class, pergroup=args.pergroup)
                density_label = torch.tensor(density_label,dtype=torch.float).to(device)
                density_label = density_label.reshape(-1,args.density_num_class)
                density_mse_label = torch.tensor(density_mse_label,dtype=torch.float).to(device).reshape(-1)

                trgt_data = trgt_data.permute(0, 2, 1)
                trgt_data_orig = trgt_data.clone()
                device = torch.device("cuda:" + str(trgt_data.get_device()) if args.cuda else "cpu")
                trgt_data, trgt_mask = mlsp.deform_input(trgt_data, lookup, args.DefRec_dist, device)
                logits_pred = model(trgt_data, activate_density_normal_ondef=True)

                loss = mlsp.calc_loss(args, logits_pred, trgt_data_orig, trgt_mask)
                trgt_print_losses['DefRec'] += loss.item() * batch_size
                trgt_mask = trgt_mask.permute(0,2,1)
                if args.Density_normal_defpart:
                    mask_cord = trgt_mask[:, :, 0]
                else:
                    mask_cord = trgt_mask[:, :, 0]*26+1
                if args.Normal_ondef:
                    normal_pred = logits_pred['Normal']
                    normal_pred = F.normalize(normal_pred,p=2,dim=-1)
                    normal_gt = F.normalize(normal_gt,p=2,dim=-1)
                    norm_loss = -torch.sum(torch.abs(torch.sum(normal_pred*normal_gt,dim=-1))*mask_cord)/(torch.sum(mask_cord))
                    norm_loss = args.normal_pred_weight*norm_loss
                    trgt_print_losses['def_normal_loss'] += norm_loss.item() * batch_size
                    loss = loss+norm_loss
                if args.Density_ondef:
                    mask_cord = mask_cord.reshape(-1)
                    density_cls_loss, density_mse_loss = mlsp.densityloss(args,logits_pred,density_mse_label,density_label,mask=mask_cord)
                    # density_loss = args.Density_weight * torch.sum(criterion1(logits_pred["density"], density_gt)*mask_cord)/(batch_size*torch.sum(mask_cord))
                    trgt_print_losses['def_density_cls_loss'] += density_cls_loss.item() * batch_size
                    trgt_print_losses['def_density_mse_loss'] += density_mse_loss.item() * batch_size
                    loss = loss+density_mse_loss+density_cls_loss
                loss.backward()

            trgt_count += batch_size


        opt.step()
        batch_idx += 1

    scheduler.step()

    # print progress
    src_print_losses = {k: v * 1.0 / src_count for (k, v) in src_print_losses.items()}
    src_acc = io.print_progress("Source", "Trn", epoch, src_print_losses)
    trgt_print_losses = {k: v * 1.0 / trgt_count for (k, v) in trgt_print_losses.items()}
    trgt_acc = io.print_progress("Target", "Trn", epoch, trgt_print_losses)

    #===================
    # Validation
    #===================
    src_val_acc, src_val_loss, src_conf_mat = test(src_val_loader, model, "Source", "Val", epoch)
    trgt_val_acc, trgt_val_loss, trgt_conf_mat = test(trgt_val_loader, model, "Target", "Val", epoch)

    # save model according to best source model (since we don't have target labels)
    if src_val_acc > src_best_val_acc:
        src_best_val_acc = src_val_acc
        src_best_val_loss = src_val_loss
        trgt_best_val_acc = trgt_val_acc
        trgt_best_val_loss = trgt_val_loss
        best_val_epoch = epoch
        best_epoch_conf_mat = trgt_conf_mat
        best_model = io.save_model(model)

io.cprint("Best model was found at epoch %d, source validation accuracy: %.4f, source validation loss: %.4f,"
          "target validation accuracy: %.4f, target validation loss: %.4f"
          % (best_val_epoch, src_best_val_acc, src_best_val_loss, trgt_best_val_acc, trgt_best_val_loss))
io.cprint("Best validtion model confusion matrix:")
io.cprint('\n' + str(best_epoch_conf_mat))

#===================
# Test
#===================
model = best_model
trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(trgt_test_loader, model, "Target", "Test", 0)
io.cprint("target test accuracy: %.4f, target test loss: %.4f" % (trgt_test_acc, trgt_best_val_loss))
io.cprint("Test confusion matrix:")
io.cprint('\n' + str(trgt_conf_mat))
