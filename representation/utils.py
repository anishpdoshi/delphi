
import io
import pysmt
import warnings


from pysmt.shortcuts import get_env, Int, BV, Real, Bool, Symbol, Ite, And, Equals
from pysmt.typing import BVType, INT, REAL, BOOL
from pysmt.smtlib.parser import SmtLibParser
from extract_oracle import convert_oracles_to_funcs

parser = SmtLibParser()

def generate_variables(num):
    return [f'p{i}' for i in range(num)]

class ExampleSimplifier(pysmt.simplifier.Simplifier):
    def __init__(self, oracle_fn_name, output, env=None):
        super().__init__(env=env)
        self.oracle_fn_name = oracle_fn_name
        self.output = output

    def walk_function(self, formula, *args, **kwargs):
        if not str(formula).split('(')[0] == self.oracle_fn_name:
            return super().walk_function(formula, *args, **kwargs)
        else:
            return self.output

def binarize(examples, smtfile, interface, oracle_fn_name=None):
    # In this case examples will be lists of full commands paired with results
    # Example ('(assert (oracle_fn 3))', False)
    with open(smtfile, 'r') as f:
        smttext = f.read()
    
    converted, oracle_map = convert_oracles_to_funcs(smttext)
    oracle_fn_name = oracle_fn_name or list(oracle_map.keys())[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        script =  parser.get_script(io.StringIO(converted))

    asserts_filter = script.filter_by_command_name('assert')
    asserts_relevant = [
        comm
        for comm in asserts_filter
        if oracle_fn_name in list(map(str, comm.args[0].get_free_variables()))
    ]
    assert len(asserts_relevant) == 1, "Multiple oracle assertions not supported yet"
    
    formula = asserts_relevant[0].args[0]
    binarized = []
    for inputs, output in examples:
        # Substitution
        input_substitution = dict(zip(interface[0], inputs))
        substituted = formula.substitute(input_substitution)
        # Simplified
        # TODO just delete entry from cache instead of recreating this obj
        simplifier = ExampleSimplifier(oracle_fn_name, output, env=get_env())
        simplified = simplifier.simplify(substituted)
        binarized.append((inputs, Bool(simplified.constant_value())))
    
    return binarized

def parse_interface(interface_string, variable_names=None):
    # (_ BitVec 32), --> (_ BitVec 32)
    if interface_string.startswith('"') and interface_string.endswith('"'):
        interface_string = interface_string[1:-1]

    def parse_type(item, var_name):
        if item.startswith('(_ BitVec'):
            arg_bv_parts = item[1:-1].split(' ')
            arg_type = BVType(width=int(arg_bv_parts[-1]))
        elif item == 'Bool':
            arg_type = BOOL
        elif item == 'Real':
            arg_type = REAL
        elif item == 'Int':
            arg_type = INT
        else:
            raise ValueError(f'could not understand type {item}')
        
        return Symbol(var_name, arg_type)

    arg_string, ret_type_string = interface_string.split(' --> ')
    
    arg_items = arg_string[1:-1].split(',')
    variable_names = variable_names or generate_variables(len(arg_items))
    
    arg_types = [
        parse_type(item, var_name)
        for item, var_name in zip(arg_items, variable_names)
        if len(item.strip()) > 0
    ]

    ret_type = parse_type(ret_type_string, 'ret_val')
    return (arg_types, ret_type)

def parse_examples_str(string, interface):
    if string.startswith('"') and string.endswith('"'):
        string = string[1:-1]
        
    examples = []
    def parse_value(value_str, value_type):
        if value_type.is_bv_type():
            return BV(int(value_str[1:-1].split(' ')[1][2:]), width=value_type.width)
        elif value_type.is_int_type():
            return Int(int(value_str))
        elif value_type.is_real_type():
            return Real(float(value_str))
        elif value_type.is_bool_type():
            return Bool(value_str == 'true')

    for example_str in string.split('|'):
        if len(example_str.strip()) == 0:
            continue
        inputs_str, output_str = example_str.split(' -> ')
        input_strs = inputs_str.split(',')
        
        input_vals = [
            parse_value(input_str, input_symb.get_type())
            for input_str, input_symb in zip(input_strs, interface[0])
            if len(input_str.strip()) > 0
        ]
        
        output_val = parse_value(output_str, interface[1].get_type())
        examples.append((input_vals, output_val))
    
    return examples

def clean_examples(examples):
    return [
        (
            [
                feat if not isinstance(feat, pysmt.fnode.FNode) else feat.constant_value()
                for feat in example[0]
            ],
            example[1] if not isinstance(example[1], pysmt.fnode.FNode) else example[1].constant_value()
        )
        for example in examples
    ]

def parse_smt(string):
    return parser.get_script(io.StringIO(f'(assert {string})')).commands[0].args[0]

def wrap_smt_representation_examples(examples, interface, smt_block):
    if isinstance(smt_block, str):
        smt_block = parse_smt(smt_block)
    
    def wrap_recur(examples):
        if len(examples) == 0:
            return smt_block
        
        example = examples[0]

        return Ite(
            And([
                Equals(input_symb, example_const)
                for input_symb, example_const in zip(interface[0], example[0])
            ]),
            example[1],
            wrap_recur(examples[1:])
        )

    return wrap_recur(examples)

if __name__ == '__main__':
    smtfile = '../examples/first_order/simple.smt2'
    interface = parse_interface('(_ BitVec 32), --> (_ BitVec 32)', ['x'])
    examples = parse_examples_str('(_ bv1 32), -> (_ bv11 32) | (_ bv2 32), -> (_ bv12 32)', interface)
    binarized = binarize(examples, smtfile, interface, oracle_fn_name='add10')
    print(binarized)


    # with open(smtfile, 'r') as f:
    #     smttext = f.read()
    
    # converted, oracle_map = convert_oracles_to_funcs(smttext)
    # assert len(oracle_map) == 1, "Multiple oracles not supported yet"
    # oracle_fn_name = list(oracle_map.keys())[0]

    # script = parser.get_script(io.StringIO(converted))

    # asserts_filter = script.filter_by_command_name('assert')
    # asserts_relevant = [
    #     comm
    #     for comm in asserts_filter
    #     if oracle_fn_name in list(map(str, comm.args[0].get_free_variables()))
    # ]
    # assert len(asserts_relevant) == 1, "Multiple oracle assertions not supported yet"
    # formula = asserts_relevant[0].args[0]
    
    # simplifier = ExtSimplifier(get_env())
    # simplified = simplifier.simplify(formula)