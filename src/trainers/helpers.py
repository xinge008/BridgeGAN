"""
 
   CC BY-NC-ND 4.0 license   
"""
import os
import torch
import torch.nn as nn
from torch.autograd import Variable, Function





# Get model list for resume
def get_model_list(dirname, key):
  if os.path.exists(dirname) is False:
    return None
  gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                os.path.isfile(os.path.join(dirname, f)) and key in f and "pkl" in f]
  if gen_models is None:
    return None
  gen_models.sort()
  last_model_name = gen_models[-1]
  return last_model_name

def _compute_true_acc(predictions):
  predictions = torch.ge(predictions.data, 0.5)
  if len(predictions.size()) == 3:
    predictions = predictions.view(predictions.size(0) * predictions.size(1) * predictions.size(2))
  acc = (predictions == 1).sum() / (1.0 * predictions.size(0))
  return acc

def _compute_fake_acc(predictions):
  predictions = torch.le(predictions.data, 0.5)
  if len(predictions.size()) == 3:
    predictions = predictions.view(predictions.size(0) * predictions.size(1) * predictions.size(2))
  acc = (predictions == 1).sum() / (1.0 * predictions.size(0))
  return acc

def _compute_true_acc2(predictions):
  _, true_predicts = torch.max(predictions.data, 1)
  acc = (true_predicts == 1).sum() / (1.0 * true_predicts.size(0))
  return acc

def _compute_fake_acc2(predictions):
  _, true_predicts = torch.max(predictions.data, 1)
  acc = (true_predicts == 0).sum() / (1.0 * true_predicts.size(0))
  return acc



CUDA = True

def feature_covariance_mat(n, d):
    ones_t = torch.ones(n).view(1, -1)
    if CUDA:
        ones_t = ones_t.cuda()

    tmp = ones_t.matmul(d)
    covariance_mat = (d.t().matmul(d) - (tmp.t().matmul(tmp) / n)) / (n)
    print covariance_mat.size()
    return covariance_mat


def forbenius_norm(mat):
    return (mat**2).sum()**0.5


class CORAL(Function):
    def forward(self, source, target):
        d = source.size()[1]
        ns, nt = source.size()[0], target.size()[0]
        cs = feature_covariance_mat(ns, source)
        ct = feature_covariance_mat(nt, target)

        self.saved = (source, target, cs, ct, ns, nt, d)

        res = forbenius_norm(cs - ct)**2/(4*d*d)
        res = torch.cuda.FloatTensor([res])

        return res

    def backward(self, grad_output):
        source, target, cs, ct, ns, nt, d = self.saved
        ones_s_t = torch.ones(ns).view(1, -1)
        ones_t_t = torch.ones(nt).view(1, -1)
        if CUDA:
            ones_s_t = ones_s_t.cuda()
            ones_t_t = ones_t_t.cuda()

        s_gradient = (source.t() - (ones_s_t.matmul(source).t().matmul(ones_s_t)/ns)).t().matmul(cs - ct) / (d*d*(ns - 1))
        t_gradient = (target.t() - (ones_t_t.matmul(target).t().matmul(ones_t_t)/nt)).t().matmul(cs - ct) / (d*d*(nt - 1))
        t_gradient = -t_gradient

        return s_gradient*grad_output, t_gradient*grad_output