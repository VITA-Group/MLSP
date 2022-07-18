import torch
import torch.nn as nn
import torch.nn.functional as F
from PointDA.model_utils import Pointnet_Encoder, Relative_Encoder, Dgcnn_Encoder
import timm
from timm.models.layers import DropPath, trunc_normal_
from knn_cuda import KNN
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from model_utils import transform_net, conv_2d, get_graph_feature, fc_layer, classifier, Group, Encoder, TransformerEncoder, PointNetFeaturePropagation
from pointnet2_ops import pointnet2_utils
from model_utils import density_classifier
K = 20


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


class PointNet(nn.Module):
    def __init__(self, args):
        super(PointNet, self).__init__()
        num_class = args.num_class
        self.args = args

        self.trans_net1 = transform_net(args, 3, 3)
        self.trans_net2 = transform_net(args, 64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        self.conv3 = conv_2d(64, 64, 1)
        self.conv4 = conv_2d(64, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)

        num_f_prev = 64 + 64 + 64 + 128

        self.C = classifier(args, num_class)
        self.DefRec = RegionReconstruction(args, num_f_prev + 1024)

    def forward(self, x, activate_DefRec=False):
        num_points = x.size(2)
        x = torch.unsqueeze(x, dim=3) #B 3 N 1

        logits = {}

        transform = self.trans_net1(x) #B 9 9
        x = x.transpose(2, 1) #B N 3 1
        x = x.squeeze(dim=3) #B N 3
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1) #B 3 N
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        transform = self.trans_net2(x2)
        x = x2.transpose(2, 1)
        x = x.squeeze(dim=3)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x4)
        x5, _ = torch.max(x, dim=2, keepdim=False)
        x = x5.squeeze(dim=2)  # batchsize*1024

        logits["cls"] = self.C(x)

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat.squeeze(dim=3), x5.repeat(1, 1, num_points)), dim=1)
            logits["DefRec"] = self.DefRec(DefRec_input)

        return logits


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        num_class = args.num_class
        self.args = args
        self.k = K

        self.input_transform_net = transform_net(args, 6, 3)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64*2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64*2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128*2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        self.bn5 = nn.BatchNorm1d(1024)
        self.conv5 = nn.Conv1d(num_f_prev, 1024, kernel_size=1, bias=False)

        self.C = classifier(args, num_class)
        self.DefRec = RegionReconstruction(args, num_f_prev + 1024)
        self.Norm_pred = Normal_prediction(args, num_f_prev + 1024)
        self.Rec_scan = RegionReconstruction(args, num_f_prev + 1024)

        self.Density_cls = Density_prediction(args, num_f_prev + 1024)
    def forward(self, x, visualization=False, activate_DefRec=False, activate_normal=False, activate_scan = False, activate_density=False,activate_density_normal_ondef=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        logits = {}

        x0 = get_graph_feature(x, self.args, k=self.k)
        transformd_x0 = self.input_transform_net(x0)
        x = torch.matmul(transformd_x0, x)

        x = get_graph_feature(x, self.args, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, self.args, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, self.args, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, self.args, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = F.leaky_relu(self.bn5(self.conv5(x_cat)), negative_slope=0.2)

        # Per feature take the point that have the highest (absolute) value.
        # Generate a feature vector for the whole shape
        x5 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x = x5

        logits["cls"] = self.C(x)
        if visualization:
            return x5

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat, x5.unsqueeze(2).repeat(1, 1, num_points)), dim=1) #bs C 1024
            logits["DefRec"] = self.DefRec(DefRec_input)
        if activate_normal:
            Normal_input = torch.cat((x_cat, x5.unsqueeze(2).repeat(1, 1, num_points)), dim=1) #bs C 1024
            logits["Normal"] = self.Norm_pred(Normal_input)
        if activate_scan:
            Rec_scan_input = torch.cat((x_cat, x5.unsqueeze(2).repeat(1, 1, num_points)), dim=1) #bs C 1024
            logits["Rec_scan"] = self.Rec_scan(Rec_scan_input)
        if activate_density:
            Density_input = torch.cat((x_cat, x5.unsqueeze(2).repeat(1, 1, num_points)), dim=1) #bs C 1024
            logits['density'], logits['density_mse'] = self.Density_cls(Density_input)

        if activate_density_normal_ondef:
            pred_input = torch.cat((x_cat, x5.unsqueeze(2).repeat(1, 1, num_points)), dim=1) #bs C 1024
            logits["DefRec"] = self.DefRec(pred_input)
            logits['density'], logits['density_mse'] = self.Density_cls(pred_input)
            logits["Normal"] = self.Norm_pred(pred_input)

        return logits


class RegionReconstruction(nn.Module):
    """
    Region Reconstruction Network - Reconstruction of a deformed region.
    For more details see https://arxiv.org/pdf/2003.12641.pdf
    """
    def __init__(self, args, input_size):
        super(RegionReconstruction, self).__init__()
        if isinstance(args,float):
            dropout = args
        else:
            dropout = args.dropout
        # self.args = args
        self.of1 = 256
        self.of2 = 256
        self.of3 = 128

        self.bn1 = nn.BatchNorm1d(self.of1)
        self.bn2 = nn.BatchNorm1d(self.of2)
        self.bn3 = nn.BatchNorm1d(self.of3)
        self.dp1 = nn.Dropout(p=dropout)
        self.dp2 = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv1d(input_size, self.of1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(self.of1, self.of2, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(self.of2, self.of3, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(self.of3, 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.dp1(F.relu(self.bn1(self.conv1(x))))
        x = self.dp2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x) #[batch_size, 3, num_points]
        return x.permute(0, 2, 1)

class Normal_prediction(nn.Module):
    """
    Normal prediction Network 
    For more details see https://arxiv.org/pdf/2003.12641.pdf
    """
    def __init__(self, args, input_size):
        super(Normal_prediction, self).__init__()
        if isinstance(args,float):
            dropout = args
        else:
            dropout = args.dropout
        # self.args = args
        self.of1 = 256
        self.of2 = 256
        self.of3 = 128

        self.bn1 = nn.BatchNorm1d(self.of1)
        self.bn2 = nn.BatchNorm1d(self.of2)
        self.bn3 = nn.BatchNorm1d(self.of3)
        self.dp1 = nn.Dropout(p=dropout)
        self.dp2 = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv1d(input_size, self.of1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(self.of1, self.of2, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(self.of2, self.of3, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(self.of3, 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.dp1(F.relu(self.bn1(self.conv1(x))))
        x = self.dp2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x) #[batch_size, 3, num_points]
        return x.permute(0, 2, 1)

class Density_prediction(nn.Module):
    def __init__(self, args, input_size):
        super(Density_prediction, self).__init__()
        if isinstance(args,float):
            dropout = args
        else:
            dropout = args.dropout
        # self.args = args
        self.of1 = 512
        # self.of2 = 256
        # self.of3 = 128

        self.bn1 = nn.BatchNorm1d(self.of1)
        # self.bn2 = nn.BatchNorm1d(self.of2)
        # self.bn3 = nn.BatchNorm1d(self.of3)
        self.dp1 = nn.Dropout(p=dropout)
        # self.dp2 = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv1d(input_size, self.of1, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(self.of1, self.of2, kernel_size=1, bias=False)
        # self.conv3 = nn.Conv1d(self.of2, self.of3, kernel_size=1, bias=False)
        # self.conv4 = nn.Conv1d(self.of3, 3, kernel_size=1, bias=False)
        self.num_class = args.density_num_class
        # self.C = density_classifier(args, self.num_class)

        activate = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = True if args.model == 'dgcnn' else False

        self.mlp1 = fc_layer(512, 256, bias=bias, activation=activate, bn=True)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.mlp2 = fc_layer(256, 256, bias=True, activation=activate, bn=True)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.mlp3 = nn.Linear(256, self.num_class)

        self.fc2 = torch.nn.Linear(self.num_class, 1, bias=False)
        for i in range(self.num_class):
            torch.nn.init.constant(self.fc2.weight[0, i], args.pergroup*i) 
        self.fc2.weight.requires_grad = False

    def forward(self, x):
        x = self.dp1(F.relu(self.bn1(self.conv1(x))))
        x = x.permute(0, 2, 1)
        # print(x.shape, self.of1)
        x = x.reshape(-1, self.of1)
        # x = self.C(x)
        x = self.dp1(self.mlp1(x))
        x2 = self.dp2(self.mlp2(x))
        logits = self.mlp3(x2)
        p_vec = F.softmax(logits, dim=1)
        # p_vec = torch.ones_like(p_vec).cuda()
        # print('self.fc2',self.fc2.weight[0])
        density = self.fc2(p_vec)
        return p_vec, density[:, 0]



class DGCNN_Propagation(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        # print('using group version 2')
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(1024, 384, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 384),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """ coor, f : B 3 G ; B C G
            coor_q, f_q : B 3 N; B C N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q

class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  config.encoder_dims
        self.encoder_type = config.encoder_type
        if self.encoder_type=='Encoder':
            self.encoder = Encoder(encoder_channel = self.encoder_dims)
        elif self.encoder_type=='Pointnet_Encoder':
            self.encoder = Pointnet_Encoder(config, self.encoder_dims)
        elif self.encoder_type=='Dgcnn_Encoder':
            self.encoder = Dgcnn_Encoder(config, self.encoder_dims)
        elif self.encoder_type=='Relative_Encoder':
            self.encoder = Relative_Encoder(encoder_channel=self.encoder_dims)
        print('------------------------our encoder type is', self.encoder_type)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        ### for deformation reconstruction
        self.propagation_2 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        self.propagation_1= PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        self.propagation_0 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

        # self.conv1 = nn.Conv1d(self.trans_dim, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, self.cls_dim, 1)

        self.DefRec = RegionReconstruction(self.config,self.trans_dim * 2 + self.trans_dim)
        
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]


        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys', logger = 'Transformer')
            print(
                get_missing_parameters_message(incompatible.missing_keys),
            )
        if incompatible.unexpected_keys:
            print('unexpected_keys', logger = 'Transformer')
            print(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
            )

        print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')


    def forward(self, pts, activate_DefRec=False):
        batch_size = pts.size(0)
        num_points = pts.size(1)
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # print('neighborhood', neighborhood.shape)
        # encoder the input cloud blocks
        if self.encoder_type=='Relative_Encoder':
            group_input_tokens = self.encoder(neighborhood, center)
        else:
            group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # print('x0', x.shape) [32,65,384]
        # transformer
        x, feature_list = self.blocks(x, pos)
        # print('x1', x.shape) [32,65,384]
        x = self.norm(x)
        # print('x2', x.shape) [32,65,384]
        concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim = -1)

        if activate_DefRec:
            feature_list = [self.norm(t)[:,1:].transpose(-1, -2).contiguous() for t in feature_list] #32 384 64
            center_level_0 = pts.transpose(-1, -2).contiguous()
            f_level_0 = center_level_0

            center_level_1 = fps(pts, 512).transpose(-1, -2).contiguous()            # 512 # B 3 N0
            f_level_1 = center_level_1
            center_level_2 = fps(pts, 256).transpose(-1, -2).contiguous()            # 256 # B 3 N1
            f_level_2 = center_level_2
            center_level_3 = center.transpose(-1, -2).contiguous()                   # 128 # B 3 G
            # print('center_level_1,center_level_2,center_level_3',center_level_1.shape,center_level_2.shape,center_level_3.shape)
            
            # init the feature by 3nn propagation
            f_level_3 = feature_list[2]
            f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1]) #B 384 N1-256 
            f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list[0]) #B 384 N2-512
            # print('f_level_1,f_level_2,f_level_3',f_level_1.shape,f_level_2.shape,f_level_3.shape)
            # bottom up
            f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2) #B 384 N1-256 
            f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1) #B 384 N2-512 
            f_level_0 =  self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1) #get point-wise feature
            # print('f_level_0,f_level_1,f_level_2',f_level_0.shape,f_level_1.shape,f_level_2.shape)
            # print('get point-wise feature', f_level_0.shape)
            
            DefRec_input = torch.cat((f_level_0, concat_f.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
            ret = self.DefRec(DefRec_input)
            return ret


        ret = self.cls_head_finetune(concat_f)
        return ret