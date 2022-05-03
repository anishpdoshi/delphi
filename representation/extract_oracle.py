import os
import re
import io
import pysmt
from pysmt.smtlib.parser import SmtLibParser

def convert_oracles_to_funcs(smttext):
    lines = smttext.split('\n')
    converted = []
    oracle_map = {}
    for line in lines:
        if not line.startswith('(declare-oracle-fun'):
            converted.append(line)
        else:
            line_no_ends = line[1:-1]
            oracle_regex = re.compile('declare\-oracle\-fun ([^\s]+) ([^\s]+) (.+)')
            match = oracle_regex.search(line_no_ends)
            if match:
                oracle_fn_name = match.group(1)
                bin_name = match.group(2)
                interface = match.group(3)
                oracle_map[oracle_fn_name] = (bin_name, interface)
                converted.append(f'(declare-fun {oracle_fn_name} {interface})')
            else:
                raise ValueError('could not parse line')
    
    return '\n'.join(converted), oracle_map

if __name__ == '__main__':
    with open('../examples/first_order/simple.smt2', 'r') as f:
        smttext = f.read()
    converted, oracle_map = convert_oracles_to_funcs(smttext)
    parser = SmtLibParser()
    script = parser.get_script(io.StringIO(converted))

