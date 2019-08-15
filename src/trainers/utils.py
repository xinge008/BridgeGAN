#coding: utf-8
import os
import torch
import numpy as np
import torch.nn as nn

from torch.autograd import Variable, Function

"""
compute the MMD loss
"""

# class Style_Loss(nn.Module):
#     def __init__(self):
#         super(Style_Loss, self).__init__()
#
#     def gram_matrix(self, x):

def gram_matrix(x):
    (b, ch, w, h) = x.size()
    features = x.view(b, ch, w*h)
    features_t = features.transpose(1,2) # (b, w*h, ch)
    gram = features.bmm(features_t)/(ch*h*w)
    return gram

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.size(0)

def compute_pairwise_distance(x, y):
    """
    :param x: a tensor of shape [num_x_samples, num_x_features]
    :param y: similar to x
    :return: a distance matrix with size [num_x_samples, num_y_samples]
    """

    norm = lambda xx: torch.sum(xx**2, dim = 1)
    return (norm(torch.unsqueeze(x, 2) - y.t())).t()


def gaussian_kernel_matrix(x, y, sigmas):
    """
    :param x: a tensor of shape [num_x_samples, num_x_features]
    :param y: similar to x
    :param sigmas: the widths of each of gaussians in the kernel
    :return: a tensor of shape [num_x_samples, num_y_samples]
    """

    beta = 1. / (2.0 * torch.unsqueeze(sigmas, 1)).float()
    dist = compute_pairwise_distance(x, y).contiguous().float()
    # print dist.size()
    s = torch.mm(beta, dist.view(1,-1))
    return torch.sum(torch.exp(-s), 0).view_as(dist)

sigmas_ = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
sigmas_np = torch.from_numpy(np.asarray(sigmas_))
sigmas_np = Variable(sigmas_np).detach().cuda(0)

def maximum_mean_discrepancy(x, y):

    kernel_x_x = gaussian_kernel_matrix(x, x, sigmas_np)
    kernel_x_y = gaussian_kernel_matrix(x, y, sigmas_np)
    kernel_y_y = gaussian_kernel_matrix(y, y, sigmas_np)

    # print kernel_x_y
    # print kernel_y_y
    # print kernel_x_y

    cost = torch.mean(kernel_x_x) + torch.mean(kernel_y_y) - 2 * torch.mean(kernel_x_y)
    # print cost
    # return torch.max(torch.FloatTensor([cost]), torch.FloatTensor([1e-4]))
    if cost.data.numpy()[0] < 1e4:
        cost.data = torch.from_numpy(np.asarray([1e-4]))
    # print cost.data.numpy()
    return cost


def difference_loss(x, y, weight = 1.0):
    """
    sub the mean value and apply the L2-Norm

    :param x: a tensor of shape: [num_samples, num_features]
    :param y: similar to x
    :param weight:
    :return:
    """
    if len(x.size())>2:
        x = x.view(x.size(0), -1)
    if len(y.size())>2:
        y = y.view(x.size(0), -1)
    x = x - torch.mean(x, 0)
    y = y - torch.mean(y, 0)
    x = nn.functional.normalize(x, p=2, dim=1)
    y = nn.functional.normalize(y, p=2, dim=1)
    correlation_matrix = torch.mm(x.t(), y)
    cost = torch.mean(correlation_matrix**2) * weight
    return cost




"""
Gradient reverse layer:
Same forward function with negative gradient during backpropagation
"""

class GradReverse(Function):
    def __init__(self, lambda_ = 1):
        self.lambda_ = lambda_

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return -grad_output * self.lambda_


# def grad_reverse(weight, x):
#     return GradReverse(0.25)(x)

grad_reverse = GradReverse(1.0)


"""
Another way to implement a gradient reverse function:
Use a registor_hook>

for example:
"""

def reverse_hook(grad):
    grad_clone = grad.clone()
    return -1. * grad_reverse

# how to call the reverse_hook function
# output = net1(input)
# h = output.register_hook(reverse_hook)
# loss.backward()


class dann_loss(nn.Module):
    def __init__(self, weight=1.0):
        super(dann_loss, self).__init__()
        self.weight = weight

    def forward(self, x, y):
        # def DANN_loss(x, y, weight=1.0):
        """
        :param x: [num_samples, num_features]
        :param y: similar to x
        :param weight: weight factor of the DANN_loss
        :return:
        """
        batch_size = x.size()[0]
        if len(x.size())>2:
            x = x.view(batch_size, -1)
            y = y.view(batch_size, -1)

        z = torch.cat((x,y), 0).float()

        # domain_label = torch.cat((all0, all1), 0)

        grl = grad_reverse(z)
        grl = nn.functional.sigmoid(grl)
        grl_x, grl_y = torch.split(grl, batch_size, 0)
        grl_x = grl_x.view(-1)
        grl_y = grl_y.view(-1)
        # print grl_y.size()
        all0 = Variable(torch.zeros(grl_x.size())).cuda()
        all1 = Variable(torch.ones(grl_y.size())).cuda()
        print all0.size()
        domain_loss_x = nn.functional.binary_cross_entropy(grl_x, all0)
        domain_loss_y = nn.functional.binary_cross_entropy(grl_y, all1)

        cost = domain_loss_x + domain_loss_y

        # if cost.data.numpy()
        return cost * self.weight


def similar_loss(x, y):
    """
    :param x: source tensor with size [num_samples, num_features]
    :param y: similar to y
    :return:  a scalar tensor representing the correlation loss value
    """
    x = x - torch.mean(x, 0)
    y = y - torch.mean(y, 0)
    x = nn.functional.normalize(x, p=2, dim=1)
    y = nn.functional.normalize(y, p=2, dim=1)
    x_c = torch.matmul(x.t(), x)
    y_c = torch.matmul(y.t(), y)

    simi_loss = torch.mean((x_c-y_c)**2)
    return simi_loss



def get_SPD(x):
    # x should be a square matrix
    h,w = x.size()
    new_x = Variable(torch.eye(h) - 1.0 / h * torch.ones(x.size())).detach().cuda()
    C = x.matmul(new_x)
    C = C.matmul(x.t())
    return C

def log_SVD(x):
    e, v = torch.symeig(x, eigenvectors=True)
    e = torch.log(e)
    e2 = Variable(torch.eye(e.size(0))).detach().cuda()
    e_mat = e2 * e.expand_as(v)
    return v.mm(e_mat).mm(v.t())

def log_geodesic_loss(x, y):
    Cs = get_SPD(x)
    Ct = get_SPD(y)

    # e,v = torch.symeig(Cs, eigenvectors=True)
    fea_cs = log_SVD(Cs)
    fea_ct = log_SVD(Ct)

    return torch.sqrt(torch.mean((fea_cs-fea_ct)**2)) # Fronbious Norm






if __name__ == '__main__':

    x = torch.from_numpy(np.random.uniform(0,1, size=(8, 128)))
    y = torch.from_numpy(np.random.uniform(0,1, size=(8, 128)))

    x = Variable(x).cuda(0)
    y = Variable(y).cuda(0)
    print compute_pairwise_distance(x, y).size()
    print maximum_mean_discrepancy(x, y)
    print similar_loss(x, y)
    print difference_loss(x, y)
    # z = grad_reverse(x)
    # print z.size()
    # print DANN_loss(x,y,weight=1)