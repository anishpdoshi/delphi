import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

import pandas as pd
from train_logics import *
import numpy as np

SS_MEAN = [2493.7060225 , 1217.44478049, 1217.61054739, 2429.59505593, 333.79507274, 2491.11564464, 55.26044223]
SS_STD = [864.40737069, 467.49305073, 461.33622201, 864.5522074 ,
       128.54888658, 834.28007298,  25.73341932]

if __name__ == '__main__':
    assert len(sys.argv) == 8
    args = [float(arg) for arg in sys.argv[1:]]
    scaler = StandardScaler()
    scaler.mean_ = np.array(SS_MEAN)
    scaler.scale_ = np.array(SS_STD)
    args_scaled = scaler.transform(np.array(args).reshape(1,-1))
    args_scaled = torch.Tensor(args_scaled)
    model = create_nn(hidden_size)
    model_filename = f'logics_nn_{hidden_size}.pt'
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    print(model(args_scaled).item())
