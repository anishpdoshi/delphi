
import torch
from train_logics import net
from nn_to_smt import layers_to_smt

if __name__ == '__main__':
    logics_net = net()
    input_vars = [f'p{i}' for i in range(7)]
    smt_func = layers_to_smt(net, input_vars)
    print(smt_func)