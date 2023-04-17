from bnd_lexer import bnd_lexer
from bnd_parser import bnd_parser
from cfg_lexer import cfg_lexer
from cfg_parser import cfg_parser

from cfg_types import *
from bnd_types import *

import sys


def generate_header(nodes):
    return f'constexpr int states_count = {len(nodes)};\n'


def generate_aggregate_function(nodes):

    aggregate_function = '''
__device__ void compute_transition_rates(float* __restrict__ transition_rates, size_t state)
{'''

    for i, node in enumerate(nodes):
        aggregate_function += f'''
    transition_rates[{i}] = {node.name}_rate(state);'''

    aggregate_function += '''
}
'''

    return aggregate_function


def generate_node_transition_fuction(node, nodes, variables):

    node_name = node.name
    up_expr = node.attributes['rate_up'].generate_code(variables, nodes, node)
    down_expr = node.attributes['rate_down'].generate_code(
        variables, nodes, node)

    return f'''
__device__ float {node_name}_rate(size_t state)
{{
    return {Id(node_name).generate_code(variables, nodes, node)} ? 
        ({down_expr}) : 
        ({up_expr});
}}
'''


def generate_kernel(bnd_stream, cfg_stream, out_file):

    bnd_program = bnd_parser.parse(bnd_stream, lexer=bnd_lexer)
    cfg_program = cfg_parser.parse(cfg_stream, lexer=cfg_lexer)

    if bnd_program is None or cfg_program is None:
        print('Error parsing input files')
        exit(1)

    # get list of nodes
    nodes = bnd_program

    # get dict of variables and their assignments
    variables = {}
    for declaration in cfg_program:
        if type(declaration) is VarDeclaration:
            variables[declaration.name] = declaration.evaluate(variables)

    f = open(out_file, "w")

    # generate header
    f.write(generate_header(nodes))

    # generate transition functions
    for node in nodes:
        f.write(generate_node_transition_fuction(node, nodes, variables))

    # generate aggregate function
    f.write(generate_aggregate_function(nodes))

    f.close()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python generator.py <bnd_file> <cfg_file> <out_file>')
        exit(1)

    bnd_file = sys.argv[1]
    cfg_file = sys.argv[2]
    out_file = sys.argv[3]

    with open(bnd_file, 'r') as bnd:
        bnd_stream = bnd.read()
    with open(cfg_file, 'r') as cfg:
        cfg_stream = cfg.read()

    generate_kernel(bnd_stream, cfg_stream, out_file)
