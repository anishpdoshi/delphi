
import sys
import itertools
import torch
import torch.nn as nn
## EAGER MODE
from torch.quantization import quantize_dynamic

ADD = '+' if len(sys.argv) == 0 or sys.argv[0] != '--bv' else 'bvadd' #+
MUL = '*' if len(sys.argv) == 0 or sys.argv[0] != '--bv' else 'bvmul' #*
DTYPE = 'Real' if len(sys.argv) == 0 or sys.argv[0] != '--bv' else '(_ BitVec 32)' #Real

def NUM(value):
    try:
        if isinstance(value, torch.Tensor):
            value = value.item()
        if value < 0:
            return f'(- {-value})'
        return f'{value}'
    except:
        return f'{value}'
    # if isinstance(value, str) or not (isinstance(value, torch.Tensor) and len(value.size()) == 0) or value >= 0:
    #     return f'{value}'
    
    # return f'(- {-value})'

def add_to_smt_recur(elements, add_op=ADD):
    if len(elements) == 1:
        return NUM(elements[0])
    return f'({add_op} {NUM(elements[0])} {add_to_smt_recur(elements[1:])})'

def matrix_mult_to_smt_recur(matrix: torch.Tensor, inputs):
    if matrix.dtype == torch.quint8:
        matrix = torch.int_repr(matrix)
    dot_products = []
    for row_i in range(matrix.shape[0]):
        elements = []
        for col_i in range(matrix.shape[1]):
            elements.append(f'({MUL} {NUM(matrix[row_i][col_i])} {NUM(inputs[col_i])})')
        dot_products.append(add_to_smt_recur(elements))
    return dot_products

# Model matrix mult w optional bias
def linear_layer_to_smt(layer: nn.Linear, inputs: torch.Tensor):
    matmul_results = matrix_mult_to_smt_recur(layer.weight, inputs)
    if layer.bias is not None:
        return [f'({ADD} {NUM(b)} {NUM(y)})' for b, y in zip(layer.bias.tolist(), matmul_results)]
    return matmul_results

def relu_to_smt(inputs: torch.Tensor):
    return [f'(ite (> {NUM(x)} 0) {NUM(x)} 0)' for x in inputs]        

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
    return created_vars, layers_smt + f'\n(assert (= {output_var} {NUM(model_output)}))'