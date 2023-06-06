import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from lib.config import cfg
from torch.nn.functional import grid_sample


BatchNorm2d = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    # 3*3的卷积层模块：在原文中使用3*3卷积模块对图像进行降维
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)


class BasicBlock(nn.Module):

    # 这个Block就是经典的卷积模块：Conv+BN+Relu
    def __init__(self, inplanes, outplanes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes )
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(outplanes, outplanes, 2*stride)

    def forward(self, x):

        out = self.conv1(x)
        # conv层与层之间需要加入BN+ReLU，以保证非线性
        out = self.bn1(out)
        out = self.relu(out)
        # 输出
        out = self.conv2(out)

        return out

class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        #print(point_features.shape, img_features.shape)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


#================addition attention (add)=======================#
class IA_Layer(nn.Module):
    def __init__(self, channels):
        print('##############ADDITION ATTENTION(ADD)#########')
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                    nn.BatchNorm1d(self.pc),
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)


    def forward(self, img_feas, point_feas):
        batch = img_feas.size(0)
        # 初始化图像和点云特征信息
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas)

        # 将图像特征和点云特征分别输入到FC层中，目的：把两者变换到一个维度上去，以便于后面融合
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)

        # 1、特征融合：图像特征和点云特征融合的方式是元素级相加的方式：即ri+rp
        # 2、然后经过tanh：将其分布变换到[-1,1]中
        # 3、FC3层变换维度，以便于后面与图像信息进行相乘
        # 4、通过sigmoid将值限制在[0,1]内，得到权重矩阵，这样图像或点云中每一个元素都有自己的权重值，我们自然也就能知道图像像素能贡献多少
        att = F.sigmoid(self.fc3(F.tanh(ri + rp))) #BNx1

        att = att.squeeze(1)
        att = att.view(batch, 1, -1) #B1N
        # print(img_feas.size(), att.size())

        # 原始图像维度变换：图像特征经过一个卷积层变换维度，使得其能够与权重矩阵进行相乘
        img_feas_new = self.conv1(img_feas)
        # 图像融合特征：用图像*权重矩阵，得到的是图像贡献的信息
        out = img_feas_new * att
        # 返回融合后图像的信息：out
        return out


class Atten_Fusion_Conv(nn.Module):
    # Li—Fusion融合函数
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()
        # 采用IA_Layer融合图像和点云信息
        self.IA_Layer = IA_Layer(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)
        # 利用IA_Layer得到融合后的图像信息
        img_features =  self.IA_Layer(img_features, point_features)
        # print("img_features:", img_features.shape)

        # 将原始点云信息和融合后的图像信息直接拼接，这样既保留了具有一定权重的图像信息（去除了不重要的图像信息），也保留了原始点云信息
        # fusion_features = img_features + point_features
        fusion_features = torch.cat([point_features, img_features], dim=1)
        # Conv+bn+relu得到最终融合结果
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """
    # 插值
    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)
    # grid_sample：torch官网提供的插值方法：
    # 	原型：torch.nn.functional.grid_sample(input,grid,mode='bilinear',padding_mode='zeros',align_corners=None)。
    # 	其中：mode为选择采样方法，有三种内插算法可选，分别是'bilinear'双线性差值、'nearest'最邻近插值、'bicubic' 双三次插值。

    # 具体参考：https://blog.csdn.net/jameschen9051/article/details/124714759
    interpolate_feature = grid_sample(feature_map, xy)  # (B,C,1,N) 我们这里采用双线性插值
    # 返回插值结果
    return interpolate_feature.squeeze(2) # (B,C,N)


def get_model(input_channels = 6, use_xyz = True):
    return Pointnet2MSG(input_channels = input_channels, use_xyz = use_xyz)


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels = 6, use_xyz = True):
        super().__init__()
        # PointNet++中的SA模块定义
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            # 调用PointNet++中的SA模块，聚合点云信息
            self.SA_modules.append(
                    PointnetSAModuleMSG(
                            npoint = cfg.RPN.SA_CONFIG.NPOINTS[k],
                            radii = cfg.RPN.SA_CONFIG.RADIUS[k],
                            nsamples = cfg.RPN.SA_CONFIG.NSAMPLE[k],
                            mlps = mlps,
                            use_xyz = use_xyz,
                            bn = cfg.RPN.USE_BN
                    )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        ##################
        # 根据cig文件中设定的参数，决定是否采用LI_FUSION模块
        # cig文件：LI_FUSION.ENABLED: True
        if cfg.LI_FUSION.ENABLED:
            # 这是图像特征抽象模块（图像降采样模块）定义
            self.Img_Block = nn.ModuleList()
            # 这是融合卷积模块定义
            self.Fusion_Conv = nn.ModuleList()
            # 反卷积模块定义
            self.DeConv = nn.ModuleList()
            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                # 下面是对Img_Block模块进行初始化：
                # 1、图像降采样模块Img_Block：根据cig文件里设定参数，将卷积层等添加到Img_Block模块中

                # 2、采用已经定义好的BasicBlock模块进行堆叠，堆叠的方法是按照：
                # 输入input:LI_FUSION.IMG_CHANNELS[i] 输出output：cfg.LI_FUSION.IMG_CHANNELS[i+1]

                # 3、cig文件中对于图像channel的设定：IMG_CHANNELS: [3, 64, 128, 256, 512]
                # 也就是说卷积层输出输出(input,output)设定应该为:(3,64),(64,128),(128,256),(256,512)
                self.Img_Block.append(BasicBlock(cfg.LI_FUSION.IMG_CHANNELS[i], cfg.LI_FUSION.IMG_CHANNELS[i+1], stride=1))
                # 根据cig文件中设定，决定是否加入图像注意力ADD_Image_Attention
                # cig文件：ADD_Image_Attention: True
                if cfg.LI_FUSION.ADD_Image_Attention:
                    # Fusion_Conv模块中添加Li-Fusion融合模块
                    self.Fusion_Conv.append(
                        Atten_Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.POINT_CHANNELS[i],
                                          cfg.LI_FUSION.POINT_CHANNELS[i]))
                else:
                    self.Fusion_Conv.append(Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + cfg.LI_FUSION.POINT_CHANNELS[i],
                                                        cfg.LI_FUSION.POINT_CHANNELS[i]))
                # 反卷积模块DeConv：通过cig设定，添加转置卷积模块ConvTranspose2d
                # cig文件：① IMG_CHANNELS: [3, 64, 128, 256, 512]  ② DeConv_Reduce: [16, 16, 16, 16]
                self.DeConv.append(nn.ConvTranspose2d(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.DeConv_Reduce[i],
                                                  kernel_size=cfg.LI_FUSION.DeConv_Kernels[i],
                                                  stride=cfg.LI_FUSION.DeConv_Kernels[i]))
            # 图像特征融合模块
            self.image_fusion_conv = nn.Conv2d(sum(cfg.LI_FUSION.DeConv_Reduce), cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, kernel_size = 1)
            # BN
            self.image_fusion_bn = torch.nn.BatchNorm2d(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4)
            # 根据ADD_Image_Attention决定是否采用图像注意力
            if cfg.LI_FUSION.ADD_Image_Attention:
                self.final_fusion_img_point = Atten_Fusion_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, cfg.LI_FUSION.IMG_FEATURES_CHANNEL, cfg.LI_FUSION.IMG_FEATURES_CHANNEL)
            else:
                self.final_fusion_img_point = Fusion_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL + cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, cfg.LI_FUSION.IMG_FEATURES_CHANNEL)

        # FP模块（逆距离加权平均）定义
        self.FP_modules = nn.ModuleList()
        # FP模块初始化
        for k in range(cfg.RPN.FP_MLPS.__len__()):
            pre_channel = cfg.RPN.FP_MLPS[k + 1][-1] if k + 1 < len(cfg.RPN.FP_MLPS) else channel_out
            self.FP_modules.append(
                    PointnetFPModule(mlp = [pre_channel + skip_channel_list[k]] + cfg.RPN.FP_MLPS[k])
            )

    def _break_up_pc(self, pc):
        # _break_up_pc函数：初始化点云参数

        # 截取点云前三个变量作为点云的xyz坐标
        xyz = pc[..., 0:3].contiguous()
        features = (
            # 截取点云pc[3:]作为点云特征
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features


    def forward(self, pointcloud: torch.cuda.FloatTensor, image=None, xy=None):
        # 点云坐标xyz及其特征features
        xyz, features = self._break_up_pc(pointcloud)
        # 将点云xyz坐标和特征升维后传给l_xyz和l_features
        l_xyz, l_features = [xyz], [features]
        # 根据cig文件，决定是否采用LI_FUSION
        # cig文件：LI_FUSION.ENABLED: True
        if cfg.LI_FUSION.ENABLED:
            #### normalize xy to [-1,1]
            size_range = [1280.0, 384.0]
            xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
            xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0  # = xy / (size_range - 1.) * 2 - 1.
            # 保存归一化后的结果
            l_xy_cor = [xy]
            img = [image]
        # 遍历SA模块：S1，S2，S3，S4（原文图2），严格意义上来讲，应该是遍历SA和图像卷积模块，因为两者数量，所以用len(self.SA_modules)
        for i in range(len(self.SA_modules)):
            # 使用SA_modules对点云特征进行提取
            li_xyz, li_features, li_index = self.SA_modules[i](l_xyz[i], l_features[i])
            # 根据cig文件，决定是否采用LI_FUSION
            if cfg.LI_FUSION.ENABLED:
                li_index = li_index.long().unsqueeze(-1).repeat(1,1,2)
                li_xy_cor = torch.gather(l_xy_cor[i],1,li_index)  # 作用：收集输入的特定维度指定位置的数值，这里就是找到对应索引的点云坐标
                # 得到图像卷积的结果
                image = self.Img_Block[i](img[i])
                #print(image.shape)
                # 图像采样器（image sampler）：对图像进行插值
                img_gather_feature = Feature_Gather(image,li_xy_cor) #, scale= 2**(i+1))
                # 采用Li-Fusion模块对图像和点云信息进行融合
                li_features = self.Fusion_Conv[i](li_features,img_gather_feature)
                # 保存数据
                l_xy_cor.append(li_xy_cor)
                # 图像每个尺度都要保存
                img.append(image)
            # 保存点云xyz
            l_xyz.append(li_xyz)
            # 保存li-fusion融合后的结果
            l_features.append(li_features)

        # FP模块：对应到原图中共有三个FP模块：P1,P2,P3（原文图2）
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
        # 下面是Image Stream的反卷积
        if cfg.LI_FUSION.ENABLED:
            #for i in range(1,len(img))
            DeConv = []
            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                # 将img作为self.DeConv的输入进行反卷积
                # 注：这里的DeConv数组和self.DeConv[i]是不同的，前者保存上采样的结果，后者是反卷积层
                DeConv.append(self.DeConv[i](img[i + 1]))
            # 将反卷积的结果进行拼接
            de_concat = torch.cat(DeConv,dim=1)

            # 将反卷积后的结果经过Conv+Bn+Relu，对应到原文是FU层
            img_fusion = F.relu(self.image_fusion_bn(self.image_fusion_conv(de_concat)))
            # 最后一次FP插值:P4（原文图2）
            img_fusion_gather_feature = Feature_Gather(img_fusion, xy)
            # 最后一次Li-Fusion融合(原文图2)
            l_features[0] = self.final_fusion_img_point(l_features[0], img_fusion_gather_feature)

        return l_xyz[0], l_features[0]


class Pointnet2MSG_returnMiddleStages(Pointnet2MSG):
    def __init__(self, input_channels = 6, use_xyz = True):
        super().__init__(input_channels, use_xyz)

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        idxs = []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            idxs.append(idx)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_xyz, l_features, idxs
