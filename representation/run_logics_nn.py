
import os
import torch
from train_logics import net
from nn_to_smt import layers_to_smt

SS_MEAN = [2493.7060225 , 1217.44478049, 1217.61054739, 2429.59505593, 333.79507274, 2491.11564464, 55.26044223]
SS_STD = [864.40737069, 467.49305073, 461.33622201, 864.5522074 ,
       128.54888658, 834.28007298,  25.73341932]

if __name__ == '__main__':
    logics_net = net()
    pretrained_path = os.path.abspath(os.path.join(__file__, '..', 'logics_nn.pt'))
    logics_net.load_state_dict(torch.load(pretrained_path))
    input_vars = [
        f'(/ (- p{i} {SS_MEAN[i]}) {SS_STD[i]})'
        for i in range(7)
    ]

    smt_func = layers_to_smt(logics_net, input_vars)
    print(smt_func)