
import sys
import itertools
import torch
import torch.nn as nn
## EAGER MODE
from torch.quantization import quantize_dynamic

ADD = '+' if len(sys.argv) == 0 or sys.argv[0] != '--bv' else 'bvadd' #+
MUL = '*' if len(sys.argv) == 0 or sys.argv[0] != '--bv' else 'bvmul' #*
DTYPE = 'Real' if len(sys.argv) == 0 or sys.argv[0] != '--bv' else '(_ BitVec 32)' #Real

def add_to_smt_recur(elements):
    if len(elements) == 1:
        return elements[0]
    return f'({ADD} {elements[0]} {add_to_smt_recur(elements[1:])})'

def matrix_mult_to_smt_recur(matrix: torch.Tensor, inputs):
    if matrix.dtype == torch.quint8:
        matrix = torch.int_repr(matrix)
    dot_products = []
    for row_i in range(matrix.shape[0]):
        elements = []
        for col_i in range(matrix.shape[1]):
            elements.append(f'({MUL} {matrix[row_i][col_i]} {inputs[col_i]})')
        dot_products.append(add_to_smt_recur(elements))
    return dot_products

# Model matrix mult w optional bias
def linear_layer_to_smt(layer: nn.Linear, inputs: torch.Tensor):
    matmul_results = matrix_mult_to_smt_recur(layer.weight, inputs)
    if layer.bias is not None:
        return [f'({ADD} {b} {y})' for b, y in zip(layer.bias.tolist(), matmul_results)]
    return matmul_results

def relu_to_smt(inputs: torch.Tensor):
    return [f'(ite (> {x} 0) {x} 0)' for x in inputs]        

# Single function flow (no intermediate vars)
def layers_to_smt(layers, inputs):
    if len(layers) == 0:
        return add_to_smt_recur(inputs)

    layer = layers[0]
   
    if isinstance(layer, nn.Linear):
        layer_results = linear_layer_to_smt(layer, inputs)
    elif isinstance(layer, nn.ReLU):
        layer_results = relu_to_smt(inputs)
    
    return layers_to_smt(layers[1:], layer_results)

def with_equality_to(layers, inputs):
    model = nn.Sequential(*layers)
    model_output = model(torch.tensor(inputs)).item()

    layers_smt, created_vars = layers_to_smt(layers)
    output_var = created_vars[-1][-1]
    return created_vars, layers_smt + f'\n(assert (= {output_var} {model_output}))'