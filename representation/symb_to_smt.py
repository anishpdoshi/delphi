from collections import namedtuple
from cvc5.pythonic import BitVec, Int, Real, Bool

Node = namedtuple("Node", ["value", "children"])
CONST = 'POS'
CONSTNEG = 'NEG'

def node_creator(op_name):
    def create(*vals):
        return Node(op_name, vals)
    return create

def parse_symb_program(prog, ops, var_names):
    parse_vars = {}
    for var_name in var_names:
        parse_vars[var_name] = Node(var_name, [])

    for op in ops:
        op_name = op if isinstance(op, str) else op.name
        if op_name not in parse_vars:
            parse_vars[op_name] = node_creator(op_name)

    return eval(prog, parse_vars)


def output_val(value, sort):
    if str(sort).startswith('BitVec'):
        if value > 0:
            return f'(_ bv{value} {sort.size()})'
        else:
            return f'(bvneg (_ bv{-value} {sort.size()}))'
    else:
        return str(value)

def output_smt(parsed, rewrite_dictionary, variable_names, sort):
    
    def output_recur(node):
        if not node:
            return ''

        if isinstance(node, Node):
            if node.value in variable_names:
                return node.value
            elif node.value.startswith(CONSTNEG):
                return output_val(-1 * int(node.value.split(CONSTNEG)[1]), sort)
            elif node.value.startswith(CONST):
                return output_val(int(node.value.split(CONST)[1]), sort)
            else:
                subterms = ' '.join([output_recur(child) for child in node.children])
                return f'({rewrite_dictionary.get(node.value, node.value)} {subterms})'
        else:
            return output_val(node, sort)
    
    return output_recur(parsed)


if __name__ == '__main__':
    prog = 'add(bvshl(p0, p1), bvshl(C-3(5), p0))'
    op_stat = namedtuple('op', ["name"])
    ops_to_use = ['add', 'bvshl']
    
    prog = prog.replace('>', 'gt').replace('<', 'lt')

    prog = re.sub(r'C([0-9])\(.+?\)', r'\1', prog)
    prog = re.sub(r'C\-([0-9])\(.+?\)', r'-\1', prog)
    print(prog)

    # for ch in re.split(r'', string)
    parsed = parse_symb_program(prog, ops_to_use, ['p0', 'p1'])
    print(parsed)
    smt = output_smt(parsed, {'add': '+'}, ['p0', 'p1'], BitVec('x', 32).sort())
    print(smt)
    # pprint(x, '')
    
    