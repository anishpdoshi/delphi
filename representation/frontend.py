import os

import torch
from nn_to_smt import layers_to_smt
from train_logics import *

import argparse
from collections import namedtuple
from cvc5.pythonic import BitVec, Bool, Int, Real

import utils
from symbolic_regression import SymbolicLearner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn representation given input/output examples')
    parser.add_argument('--examples', type=str, help='Stringified input/output oracle assumptions.')
    parser.add_argument('--interface', type=str, help='Oracle interface')
    parser.add_argument('--type', choices=['dt', 'symb_regr', 'neural'], help='The type of representation to learn.')
    parser.add_argument('--pretrained', type=str, default=None, help='If given, the prefix of the pretrained file')
    # parser.add_argument('--logics', action='store_true', help='Whether or not to use pretrained LOGiCs models')
    # parser.add_argument('--sort', choices=['bv', 'int'], help='The types of logics to support')
    args = parser.parse_args()

    if args.type == 'neural':
        # just print the logics net as SMT for now
        # learner = NeuralLearner()
        SS_MEAN = [2493.7060225 , 1217.44478049, 1217.61054739, 2429.59505593, 333.79507274, 2491.11564464, 55.26044223]
        SS_STD = [864.40737069, 467.49305073, 461.33622201, 864.5522074 , 128.54888658, 834.28007298,  25.73341932]

        logics_net = create_nn(hidden_size)
        model_filename = f'logics_nn_{hidden_size}.pt'
        pretrained_path = os.path.abspath(os.path.join(__file__, '..', model_filename))
        logics_net.load_state_dict(torch.load(pretrained_path))
        input_vars = [f'(/ (- p{i} {SS_MEAN[i]}) {SS_STD[i]})' for i in range(logics_net[0].in_features)]

        smt_func = layers_to_smt(logics_net, input_vars)
        print(smt_func)
        exit(0)

    elif args.type == 'dt':
        learner = DecisionTreeLearner(interface)
    elif args.type == 'symb_regr':
        if args.interface:
            interface = utils.parse_interface(args.interface)
        if args.examples:
            examples = utils.parse_examples_str(args.examples, interface)
        learner = SymbolicLearner(interface)
    else:
        if args.interface:
            interface = utils.parse_interface(args.interface)
        if args.examples:
            examples = utils.parse_examples_str(args.examples, interface)
        learner = SymbolicLearner(interface)

    if args.pretrained is not None:
        learner.load(args.pretrained)
    
    learner.train(examples)

    print('--------SMT----------')
    print(learner.to_smt2())
