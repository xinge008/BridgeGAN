import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable, Function

from utils import *

a = dann_loss(weight=1.0)

if __name__ == '__main__':
    x = Variable(torch.from_numpy(np.random.uniform(0,1, size=(8, 128)))).cuda()
    y = Variable(torch.from_numpy(np.random.uniform(0,1, size=(8, 128)))).cuda()
    b = a(x,y)
    print b