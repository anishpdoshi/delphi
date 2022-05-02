
from cvc5.pythonic import *

def generate_variables(num):
    return [f'p{i}' for i in range(num)]

def parse_interface(interface_string):
    # ((_ BitVec 32),) -> (_ BitVec 32)
    if interface_string.startswith('"') and interface_string.endswith('"'):
        interface_string = interface_string[1:-1]


    def parse_type(item, var_name):
        if item.startswith('(_ BitVec'):
            arg_bv_parts = item[1:-1].split(' ')
            arg_type = BitVec(var_name, int(arg_bv_parts[-1]))
        elif item == 'Bool':
            arg_type = Bool(var_name)
        elif item == 'Real':
            arg_type = Real(var_name)
        elif item == 'Int':
            arg_type = Int(var_name)
        else:
            raise ValueError(f'could not understand type {item}')
        
        return arg_type

    arg_string, ret_type_string = interface_string.split(' -> ')
    
    arg_items = arg_string[1:-1].split(',')
    variable_names = generate_variables(len(arg_items))
    arg_types = []

    for item, var_name in zip(arg_items, variable_names):
        if len(item.strip()) > 0:
            arg_types.append(parse_type(item, var_name))

    ret_type = parse_type(ret_type_string, 'ret_val')
    return (arg_types, ret_type)

def parse_examples_str(string, interface):
    if string.startswith('"') and string.endswith('"'):
        string = string[1:-1]
        
    examples = []
    def parse_value(value_str, value_type):
        type_str = str(value_type.sort())
        if type_str.startswith('BitVec'):
            # re.match(r'\(_ bv([0-9]+) [0-9]+\)', value_str).group(1)
            return int(value_str[1:-1].split(' ')[1][2:])
        elif type_str == 'Int':
            return int(value_str)
        elif type_str == 'Real':
            return float(value_str)
        elif type_str == 'Bool':
            return value_str == 'true'

    for example_str in string.split('|'):
        if len(example_str.strip()) == 0:
            continue
        inputs_str, output_str = example_str.split(' -> ')
        input_strs = inputs_str.split(',')
        
        input_vals = []
        for input_str, input_type in zip(input_strs, interface[0]):
            if len(input_str.strip()) > 0:
                input_vals.append(parse_value(input_str, input_type))
        output_val = parse_value(output_str, interface[1])
        examples.append((input_vals, output_val))
    return examples