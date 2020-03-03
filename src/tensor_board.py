from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import torchvision.transforms.functional as TF
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
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

factor = 8

def computeTime(model, device='cpu'):
    inputs = torch.randn(1, 3, 512,512)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()
        writer.add_graph(model, inputs)
        writer.close()

    # module = torch.jit.trace(model, inputs)
    #m = torch.jit.script(model)
    #torch.jit.save(m,'test.pt')
    model.eval()

    i = 0
    time_spent = []

    while i < 20:
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    print('Avg execution time (ms): {:.3f}'.format(np.mean(time_spent)))



def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  print(next(model.parameters()).device)
  model.to("cuda")
  summary(model, (3, 512, 512),device="cuda")

  computeTime(model,"cuda")

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)