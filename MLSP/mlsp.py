import numpy as np
import torch
import utils.pc_utils as pc_utils
import random
import pcl

DefRec_SCALER = 20.0


def deform_input(X, lookup, DefRec_dist='volume_based_voxels', device='cuda:0',groups=1):
    """
    Deform a region in the point cloud. For more details see https://arxiv.org/pdf/2003.12641.pdf
    Input:
        args - commmand line arguments
        X - Point cloud [B, C, N]
        lookup - regions center point
        device - cuda/cpu
    Return:
        X - Point cloud with a deformed region
        mask - 0/1 label per point indicating if the point was centered
    """
    
    # get points' regions
    regions = pc_utils.assign_region_to_point(X, device)

    n = pc_utils.NREGIONS
    min_pts = 40
    region_ids = np.random.permutation(n ** 3)
    mask = torch.zeros_like(X).to(device)  # binary mask of deformed points

    for b in range(X.shape[0]):
        iters=0
        if DefRec_dist == 'volume_based_radius':
            X[b, :, :], indices = pc_utils.collapse_to_point(X[b, :, :], device)
            mask[b, :3, indices] = 1
        else:
            for i in region_ids:
                # print(iters,torch.sum(mask))
                ind = regions[b, :] == i
                # if there are enough points in the region
                if torch.sum(ind) >= min_pts:
                    iters = iters+1
                    region = lookup[i].cpu().numpy()  # current region average point
                    mask[b, :3, ind] = 1
                    num_points = int(torch.sum(ind).cpu().numpy())
                    if DefRec_dist == 'volume_based_voxels':
                        rnd_pts = pc_utils.draw_from_gaussian(region, num_points)
                        X[b, :3, ind] = torch.tensor(rnd_pts, dtype=torch.float).to(device)
                    if iters>=groups:
                        break  # move to the next shape in the batch
    return X, mask


def scan_input(X, device, pixel_size = 0.07):
    # X - Point cloud [B, N, C]
    pixel_size = random.uniform(0.045,0.075)
    mask = torch.zeros_like(X).to(device)  # binary mask of deformed points
    for b in range(X.shape[0]):
        origin_pc = X[b].cpu().numpy()
        pc, single_mask = p_scan(origin_pc, pixel_size)
        X[b] = torch.tensor(pc,dtype=torch.float32)
        mask[b] = torch.tensor(single_mask)
    return X, mask

def p_scan(pc, pixel_size):
    pixel = int(2 / pixel_size)
    rotated_pc = rotate_point_cloud_3d(pc)
    pc_compress = (rotated_pc[:,2] + 1) / 2 * pixel * pixel + (rotated_pc[:,1] + 1) / 2 * pixel

    points_list = [None for i in range((pixel + 5) * (pixel + 5))]
    pc_compress = pc_compress.astype(np.int)
    mask = np.ones_like(pc)
    scan_points = np.zeros_like(pc)
    for index, point in enumerate(rotated_pc):
        compress_index = pc_compress[index]
        if compress_index > len(points_list):
            print('out of index:', compress_index, len(points_list), point, pc[index], (pc[index] ** 2).sum(), (point ** 2).sum())
        if points_list[compress_index] is None:
            points_list[compress_index] = index
        elif point[0] > rotated_pc[points_list[compress_index]][0]:
            points_list[compress_index] = index
    points_list = list(filter(lambda x:x is not None, points_list))
    mask[points_list,:3] = 0.0
# =============================================================================
#     points_list = pc[points_list]
#     return points_list
# =============================================================================
    scan_points[points_list] = pc[points_list]
    return scan_points, mask

def drop_hole(pc, p):
    random_point = np.random.randint(0, pc.shape[0])
    index = np.linalg.norm(pc - pc[random_point].reshape(1,3), axis=1).argsort()
    return pc[index[int(pc.shape[0] * p):]]

def rotate_point_cloud_3d(pc):
    rotation_angle = np.random.rand(3) * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix_1 = np.array([[cosval[0], 0, sinval[0]],
                                [0, 1, 0],
                                [-sinval[0], 0, cosval[0]]])
    rotation_matrix_2 = np.array([[1, 0, 0],
                                [0, cosval[1], -sinval[1]],
                                [0, sinval[1], cosval[1]]])
    rotation_matrix_3 = np.array([[cosval[2], -sinval[2], 0],
                                 [sinval[2], cosval[2], 0],
                                 [0, 0, 1]])
    rotation_matrix = np.matmul(np.matmul(rotation_matrix_1, rotation_matrix_2), rotation_matrix_3)
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def chamfer_distance(p1, p2, mask):
    """
    Calculate Chamfer Distance between two point sets
    Input:
        p1: size[B, C, N]
        p2: size[B, C, N]
    Return:
        sum of all batches of Chamfer Distance of two point sets
    """

    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)
    # print(p1.shape) [B, N, C]
    # add dimension
    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    # repeat point values at the new dimension
    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    # calc norm between each point in p1 and each point in p2
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3) ** 2

    # add big value to points not in voxel and small 0 to those in voxel
    mask_cord = mask[:, :, 0]  # take only one coordinate  (batch_size, #points)
    m = mask_cord.clone()
    m[m == 0] = 100  # assign big value to points not in the voxel
    m[m == 1] = 0
    m = m.view(dist.size(0), 1, dist.size(2))  # transform to (batch_size, 1, #points)
    dist = dist + m

    # take the minimum distance for each point in p1 and sum over batch
    dist = torch.min(dist, dim=2)[0]  # for each point in p1 find the min in p2 (takes only from relevant ones because of the previous step)
    # print('dist.shape',dist.shape,index.shape,mask_cord.shape) #[B, N]
    sum_pc = torch.sum(dist * mask_cord, dim=1)  # sum distances of each example (array broadcasting - zero distance of points not in the voxel for p1 and sum distances)
    dist = torch.sum(torch.div(sum_pc, torch.sum(mask_cord, dim=1)))  # divide each pc with the number of active points and sum
    return dist


def reconstruction_loss(pred, gold, mask):
    """
    Calculate symmetric chamfer Distance between predictions and labels
    Input:
        pred: size[B, C, N]
        gold: size[B, C, N]
        mask: size[B, C, N]
    Return:
        mean batch loss
    """
    gold = gold.clone()

    batch_size = pred.size(0)

    # [batch_size, #points, coordinates]
    gold = gold.permute(0, 2, 1)
    mask = mask.permute(0, 2, 1)

    # calc average chamfer distance for each direction
    dist_gold = chamfer_distance(gold, pred, mask)
    dist_pred = chamfer_distance(pred, gold, mask)
    chamfer_loss = dist_gold + dist_pred

    # average loss
    loss = (1 / batch_size) * chamfer_loss

    return loss

def findindexs(pred, gold, mask):

    gold = gold.clone()

    gold = gold.permute(0, 2, 1)
    mask = mask.permute(0, 2, 1)

    index1 = findneareat_index(pred, gold,mask)
    index2 = findneareat_index(gold, pred,mask)
    return [index1, index2]


def findneareat_index(p1,p2,mask):
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)
    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    # repeat point values at the new dimension
    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    # calc norm between each point in p1 and each point in p2
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3) ** 2

    # add big value to points not in voxel and small 0 to those in voxel
    mask_cord = mask[:, :, 0]  # take only one coordinate  (batch_size, #points)
    m = mask_cord.clone()
    m[m == 0] = 100  # assign big value to points not in the voxel
    m[m == 1] = 0
    m = m.view(dist.size(0), 1, dist.size(2))  # transform to (batch_size, 1, #points)
    dist = dist + m

    # take the minimum distance for each point in p1 and sum over batch
    _,index = torch.min(dist, dim=2)
    return index

def calc_loss(args, logits, labels, mask):
    """
    Calc. DefRec loss.
    Return: loss
    """
    prediction = logits['DefRec']
    loss = args.DefRec_weight * reconstruction_loss(prediction, labels, mask) * DefRec_SCALER
    return loss

def calc_scan_loss(args, logits, labels, mask):
    """
    Calc. DefRec loss.
    Return: loss
    """
    prediction = logits['Rec_scan']
    loss = args.Scan_Rec_weight * reconstruction_loss(prediction, labels, mask) * DefRec_SCALER
    return loss

def cal_density(batch_pts, radius, num_cls, pergroup = 2, shift=0,K=100):
    batch_size, num_points, _ = batch_pts.size()
    batch_cls = []
    batch_val = []
    for i in range(batch_size):
        pts = batch_pts[i].cpu().numpy()
        cloud = pcl.PointCloud()
        cloud.from_array(np.array(pts, dtype=np.float32))
        kdtree = cloud.make_kdtree_flann()
        searchPoint = pcl.PointCloud()
        searchPoint.from_array(np.array(pts, dtype=np.float32))

        [ind, sqdist] = kdtree.radius_search_for_cloud(searchPoint, radius,K)
        ind = np.array(ind)
        row = np.array((ind!=0).sum(1))
        row = row-shift
        row[row<0]=0
        row[row>((num_cls-1)*pergroup)]=(num_cls-1)*pergroup
        # print('row',row,row.shape)
        cls1 = np.floor((row)/pergroup).astype(np.int32)
        cls1 = np.identity(num_cls)[cls1]
        cls2 = np.ceil((row)/pergroup).astype(np.int32)
        cls2 = np.identity(num_cls)[cls2]
        cls = (cls1+cls2)/2.0
        # print('cls',cls,cls.shape)
        batch_cls.append(cls)
        batch_val.append(row)

        # bar[k][bar[k]<10]=10
        # bar[k][bar[k]>(pergroup*(class_num-1)+10)] = pergroup*(class_num-1)+10
        # bar[k] = np.floor(np.array(bar[k]-10)/pergroup)
        # bar[k]=list(bar[k])
    return np.array(batch_cls), np.array(batch_val)

import torch.nn.functional as F
def normal_prediction_loss(pred, gt):
    # pred/gt normal direction [batch_size, num_points, 3]
    batch_size, num_points, _ = pred.size()
    pred = F.normalize(pred,p=2,dim=-1)
    gt = F.normalize(gt,p=2,dim=-1)
    loss = -torch.sum(torch.abs(torch.sum(pred*gt,dim=-1)))/(batch_size*num_points)
    # pred[torch.where(pred[:,:,2]*gt[:,:,2]<0)] = -pred[torch.where(pred[:,:,2]*gt[:,:,2]<0)]
    # loss = -torch.sum(pred*gt)/(batch_size*num_points)
    return loss

def calc_normal_loss(args, prediction, labels):
    loss = args.normal_pred_weight * normal_prediction_loss(prediction, labels)
    return loss

def calc_def_normal_loss(args, logits, normal_labels, mask, indexes,device,all=False):
    mask = mask.permute(0, 2, 1)
    # mask_cord = mask[:, :, 0]
    if args.Density_normal_defpart:
        mask_cord = mask[:, :, 0]
    else:
        mask_cord = mask[:, :, 0]*26+1

    index1, index2 = indexes
    batch_size=index1.size(0)
    num_points = index1.size(1)

    normal_pred = logits['Normal']
    normal_pred = F.normalize(normal_pred,p=2,dim=-1)
    normal_labels = F.normalize(normal_labels,p=2,dim=-1)
    # print('index1',index1)
    # print('index2',index2)

    normal_gt=[]
    for i in range(batch_size):
        normal_gt.append(torch.index_select(normal_labels[i],0,index1[i]))
    normal_gt = torch.stack(normal_gt).to(device)
    # print('normal_gt',normal_gt.shape)
    #torch.sum(torch.div(sum_pc, torch.sum(mask_cord, dim=1)))
    temp = torch.abs(torch.sum(normal_pred*normal_gt,dim=-1)) #[B, N]

    sum_loss = torch.sum(temp * mask_cord, dim=1)
    loss = -torch.sum(torch.div(sum_loss, torch.sum(mask_cord, dim=1)))/batch_size
    # loss = -torch.sum(temp)/(batch_size*num_points)


    pred_gt = []
    for i in range(batch_size):
        pred_gt.append(torch.index_select(normal_pred[i],0,index2[i]))
    pred_gt = torch.stack(pred_gt).to(device)
    tmp = torch.abs(torch.sum(pred_gt*normal_labels,dim=-1))
    sum_loss = torch.sum(tmp * mask_cord, dim=1)
    loss = loss-torch.sum(torch.div(sum_loss, torch.sum(mask_cord, dim=1)))/batch_size
    # print('normal_labels',normal_labels,'normal_gt',normal_gt)
    # print('normal_pred',normal_pred,'pred_gt',pred_gt)
    return args.normal_pred_weight * loss

def calc_def_density_loss(args, logits, density_labels, mask, indexes, device, criterion,all=False):
    mask = mask.permute(0, 2, 1)
    mask_cord = mask[:, :, 0].reshape(-1)
    index1, index2 = indexes
    batch_size=index1.size(0)
    num_points = index1.size(1)

    origin_density_pred = logits["density"].clone()
    density_pred = logits["density"].reshape(batch_size, num_points, args.density_num_class)
    origin_density_labels = density_labels.clone()
    density_labels = density_labels.reshape(batch_size, num_points, 1)

    density_gt=[]
    for i in range(batch_size):
        density_gt.append(torch.index_select(density_labels[i],0,index1[i]))
    density_gt = torch.stack(density_gt).to(device)
    # print('density_gt',density_gt.shape)
    density_gt = density_gt.reshape(-1)
    tmp = criterion(origin_density_pred, density_gt)
    # print('temp',tmp.shape)
    if all:
        loss = args.Density_weight * torch.sum(tmp)/(batch_size*num_points)
    else:
        loss = args.Density_weight * torch.sum(tmp*mask_cord)/(batch_size*torch.sum(mask_cord))

    pred_gt = []
    for i in range(batch_size):
        pred_gt.append(torch.index_select(density_pred[i],0,index2[i]))
    pred_gt = torch.stack(pred_gt).to(device)
    # print('pred_gt',pred_gt.shape)
    pred_gt = pred_gt.reshape(-1,args.density_num_class)
    tmp = criterion(pred_gt, origin_density_labels)
    # loss = loss+args.Density_weight *
    if all:
        loss = loss + args.Density_weight * torch.sum(tmp)/(batch_size*num_points)
    else:
        loss = loss + args.Density_weight * torch.sum(tmp*mask_cord)/(batch_size*torch.sum(mask_cord))
    return loss

def deform_densityloss(args, logits, density_labels, density_mse_label,mask, indexes, device):
    # density_labels : [B*N, C]
    # density_mse_label : [B, N]
    mask = mask.permute(0, 2, 1)
    if args.Density_normal_defpart:
        mask_cord = mask[:, :, 0]
    else:
        mask_cord = mask[:, :, 0]*26+1

    mask_cord = mask_cord.reshape(-1)
    index1, index2 = indexes
    batch_size=index1.size(0)
    num_points = index1.size(1)

    origin_density_pred = logits["density"].clone() #[B*N,C]
    density_pred = logits["density"].reshape(batch_size, num_points, args.density_num_class)
    origin_density_mse_pred = logits["density_mse"].clone() #[B*N]
    density_mse_pred = logits["density_mse"].reshape(batch_size, num_points)

    origin_density_labels = density_labels.clone()
    density_labels = density_labels.reshape(batch_size, num_points, args.density_num_class)

    density_gt=[]
    density_mse_gt = []
    for i in range(batch_size):
        density_gt.append(torch.index_select(density_labels[i],0,index1[i]))
        density_mse_gt.append(torch.index_select(density_mse_label[i],0,index1[i]))
    density_gt = torch.stack(density_gt).to(device)
    density_mse_gt = torch.stack(density_mse_gt).to(device)
    # print('density_gt',density_gt.shape) #[B, N, C]
    # print('density_mse_gt',density_mse_gt.shape) #[B, N]
    density_gt = density_gt.reshape(-1,args.density_num_class)
    density_mse_gt = density_mse_gt.reshape(-1)

    kl, mse = densityloss(args,logits,density_mse_gt,density_gt,mask=mask_cord)

    pred_gt = []
    pred_mse_gt = []
    for i in range(batch_size):
        pred_gt.append(torch.index_select(density_pred[i],0,index2[i]))
        pred_mse_gt.append(torch.index_select(density_mse_pred[i],0,index2[i]))
    pred_gt = torch.stack(pred_gt).to(device)
    pred_mse_gt = torch.stack(pred_mse_gt).to(device)
    # print('pred_gt',pred_gt.shape)
    pred_gt = pred_gt.reshape(-1,args.density_num_class)
    pred_mse_gt = pred_mse_gt.reshape(-1)

    logits={}
    logits['density'] = origin_density_labels
    logits['density_mse'] = density_mse_label.reshape(-1)
    kl1, mse1 = densityloss(args,logits,pred_mse_gt,pred_gt,mask=mask_cord)
    # tmp = criterion(pred_gt, origin_density_labels)
    # # loss = loss+args.Density_weight *
    # if all:
    #     loss = loss + args.Density_weight * torch.sum(tmp)/(batch_size*num_points)
    # else:
    #     loss = loss + args.Density_weight * torch.sum(tmp*mask_cord)/(batch_size*torch.sum(mask_cord))
    return kl+kl1, mse+mse1


def densityloss(args,logits,target,target_vec,mask=None):
    lambda_1=0.05
    lambda_2=1

    p_vec = logits['density']
    # p_vec target_vec : [N,C]
    # mask : [N]
    # target: [N]
    # p_val: [N]
    if mask is not None:
        tmp = torch.sum(target_vec * torch.log(p_vec + 1e-10), dim=1)
        kl = -args.Density_weight * torch.sum(tmp*mask)/torch.sum(mask)*lambda_2
        # kl = -1*torch.sum(target_vec * torch.log(p_vec + 1e-10), dim=1)*mask
    else:
        # num = p_vec.size(0)
        tmp = torch.sum(target_vec * torch.log(p_vec + 1e-10), dim=1)
        kl = -args.Density_weight * torch.mean(tmp)*lambda_2

    p_val = logits['density_mse']
    if mask is not None:
        tmp = F.l1_loss(p_val,target,reduction='none')
        mae = args.Density_weight * torch.sum(tmp*mask)/torch.sum(mask)*lambda_1
    else:
        mae = args.Density_weight * F.l1_loss(p_val, target)*lambda_1
    return kl, mae






def calc_loss_ptrans(args, logits, labels, mask):
    """
    Calc. DefRec loss.
    Return: loss
    """
    prediction = logits
    loss = args.DefRec_weight * reconstruction_loss(prediction, labels, mask) * DefRec_SCALER
    return loss



# class KLLoss(torch.nn.Module):
#     def __init__(self):
#         super(KLLoss, self).__init__()

#     def forward(self, pred, labels):
#         rce = torch.mean(-1*torch.sum(labels * torch.log(pred + 1e-10), dim=1))
#         return rce

# class DensityLoss(nn.Module):

#     def __init__(self, lambda_1=0.01, lambda_2=1):
#         super().__init__()
#         self.lambda_1 = lambda_1
#         self.lambda_2 = lambda_2
#         self.crti_kl = KLLoss()

#     def forward(self, args, logits, p_vec, age, target, target_vec):

#         target = target.type(torch.FloatTensor).cuda()

#         # mean loss
#         mse = (age - target)**2
#         mean_loss = mse.mean() / 2.0
#         mae = F.l1_loss(age, target)

#         # kl loss
#         kl = self.crti_kl(p_vec, target_vec)

#         losses = dict(mean_loss=self.lambda_1 * mae,
#                       kl_loss=self.lambda_2 * kl)
#         return losses