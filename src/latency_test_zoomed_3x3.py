from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
import time
import numpy as np
from models.networks.res2net_dcn_zoomed_conv import zoomedConv3x3, conv3x3old,USConv2d
from torchsummary import summary


def computeTime(model,filter_size,size, device='cpu'):
    inputs = torch.randn(10, filter_size, size, size)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    #module = torch.jit.trace(model, inputs)
    #m = torch.jit.script(model)
    #torch.jit.save(m,'test.pt')
    model.eval()

    i = 0
    time_spent = []
    while i < 2:
        start_time = time.time()
        with torch.no_grad():
            a = model(inputs)
            print(a.size())

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    print('Avg execution time (ms) running in {}: {:.3f}'.format(device, np.mean(time_spent)))



def main():

  
  print('Creating model...')
  filter_size = 20
  size = 500
  model = conv3x3old(filter_size, filter_size, 3)
  model.to("cpu")
  summary(model, (filter_size, size, size), device="cpu")
  computeTime(model, filter_size,size, 'cpu')
  computeTime(model, filter_size,size, 'cuda')


  model = USConv2d(filter_size, filter_size,3,padding=1)
  model.to("cpu")
  summary(model, (filter_size, size, size), device="cpu")
  computeTime(model,filter_size,size, 'cpu')
  computeTime(model,filter_size,size, 'cuda')


  model = zoomedConv3x3(filter_size, filter_size)
  print(next(model.parameters()).device)
  model.to("cpu")
  summary(model, (filter_size, size, size),device="cpu")
  computeTime(model,filter_size,size,'cpu')
  computeTime(model,filter_size,size,'cuda')




if __name__ == '__main__':
  main()