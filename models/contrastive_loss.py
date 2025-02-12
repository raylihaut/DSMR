import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models
import torch.nn.functional as F
import numpy as np
import math

# CVPR 2018 Version 
class smooth_loss(nn.Module):
    def __init__(self):
        super(smooth_loss, self).__init__()

    def __call__(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0


class recon_loss(nn.Module):
    def __init__(self,):
        super(recon_loss, self).__init__()

    def __call__(self, x, y):
        return torch.mean( (x - y) ** 2 )
        

# MICCAI 2018 Version 
class kl_loss(nn.Module):
    def __init__(self):
        super(kl_loss, self).__init__()
        # self.image_sigma = 0.02
        # self.prior_lambda = 10.0
        self.image_sigma = 0.01
        self.prior_lambda = 25.0

    def __call__(self, flow_mean, flow_sigma):
        # print(1, flow_mean.sum())
        # print(2, flow_sigma.sum())
        ndims = len(flow_mean.shape[2:])

        D = self._degree_matrix(flow_mean.shape[2:]).cuda()
        # print(3, D.sum())

        # sigma terms
        sigma_term = torch.mean(self.prior_lambda * D * torch.exp(flow_sigma.cuda()) - flow_sigma.cuda())
        # print(4, sigma_term.sum())

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda * self.prec_loss(flow_mean)
        # print(5, self.prior_lambda, prec_term)

        # combine terms
        return 0.5 * ndims * (sigma_term + prec_term)  # ndims because we averaged over dimensions as well

    def _adj_filt(self, ndims):
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied
        # ith feature to ith feature
        filt = np.zeros([ndims, ndims] + [3] * ndims)
        for i in range(ndims):
            filt[i, i, ...] = filt_inner

        return filt

    def _degree_matrix(self, vol_shape):
        # get shape stats
        ndims = len(vol_shape)

        conv_fn = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        conv_fn.weight = torch.nn.Parameter(torch.from_numpy(self._adj_filt(ndims)).float())
        conv_result = conv_fn(torch.ones([1] + [ndims, *vol_shape]))

        return conv_result

    def prec_loss(self, y_pred):
        vol_shape = y_pred.shape[2:]
        ndims = len(vol_shape)

        x = y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]
        x2 = torch.mean(torch.mul(x, x))

        y = y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]
        y2 = torch.mean(torch.mul(y, y))

        z = y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]
        z2 = torch.mean(torch.mul(z, z))

        return 0.5 * torch.add(torch.add(x2, y2), z2) / ndims


class lamda_mse_loss(nn.Module):
    def __init__(self):
        super(lamda_mse_loss, self).__init__()
        self.image_sigma=0.01

    def __call__(self, x, y):
        return 1.0 / (self.image_sigma ** 2) * torch.mean( (x - y) ** 2 )

class contrastive_loss(nn.Module):

    def __init__(self, batch_size=1, temperature=0.5, use_cosine_similarity=True):
        super(contrastive_loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.cuda()

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def __call__(self, zis, zjs):
        # print(1, zjs.shape)
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)    
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)   #正样本的得分
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)  #负样本的得分

        logits = torch.cat((positives, negatives), dim=1)   #将正样本和负样本在列上cat起来之后的值
        logits /= self.temperature
        # print('logits:', logits)

        labels = torch.zeros(2 * self.batch_size).cuda().long()  
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

class Contrastive_Loss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
    def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size , temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
   

        # Normalize the representations
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute the similarity matrix
        #batch_size = z1.shape[0]
        representations = torch.cat([z1, z2], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        #similarity_matrix = torch.matmul(z1, z2.T) / self.temperature

        # Create labels
        #labels = torch.arange(batch_size, device=device)
        labels = torch.zeros(2 * self.batch_size).cuda().long()
        
        # Calculate loss for both directions
        loss_1 = self.criterion(similarity_matrix, labels)
        loss_2 = self.criterion(similarity_matrix.T, labels)

        # Combine losses
        loss = (loss_1 + loss_2) / 2

        return loss
