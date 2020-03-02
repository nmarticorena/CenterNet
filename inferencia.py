import time
import numpy as np
import torch
def computeTime(model, device='cuda'):
    inputs = torch.randn(10, 3, 224, 224)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    i = 0
    time_spent = []
    while i < 100:
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    print('Avg execution time (ms): {:.3f}'.format(np.mean(time_spent)))
