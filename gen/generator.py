from bnd_lexer import bnd_lexer
from bnd_parser import bnd_parser
from cfg_lexer import cfg_lexer
from cfg_parser import cfg_parser

from cfg_types import *
from bnd_types import *

import sys
import os
import json
import argparse


def get_internals(nodes, cfg):
    internals = []

    for decl in cfg:
        if type(decl) == AttrDeclaration:
            # TODO here we expect that is_internal expression is a constant
            if decl.attr == 'is_internal' and decl.expr.evaluate({}) == True:
                internals.append(decl.name)

    return internals


def get_initial_states(nodes, cfg):
    initials = {}

    for decl in cfg:
        if type(decl) == AttrDeclaration:
            # TODO here we expect that isstate expression is a constant
            if decl.attr == 'istate':
                initials[decl.name] = decl.expr.evaluate({}) != 0

    return initials


def get_constant(name, cfg):
    for decl in cfg:
        if type(decl) == ConstantDeclaration:
            if decl.name == name:
                return decl.expr.evaluate({})

    return None


def generate_tr_header_file(nodes, cfg):
    return f'''#pragma once
constexpr int states_count = {len(nodes)};
'''


def generate_config(nodes, cfg, variables):
    internals = get_internals(nodes, cfg)
    initials = get_initial_states(nodes, cfg)
    max_time = get_constant('max_time', cfg)
    time_tick = get_constant('time_tick', cfg)
    seed = get_constant('seed_pseudorandom', cfg)
    discrete_time = get_constant('discrete_time', cfg)
    sample_count = get_constant('sample_count', cfg)

    return {
        "max_time": max_time if max_time is not None else 10,
        "time_tick": time_tick if time_tick is not None else 1,
        "seed": int(seed) if seed is not None else 0,
        "discrete_time": int(discrete_time) if discrete_time is not None else 0,
        "sample_count": sample_count if sample_count is not None else 1000000,
        "nodes": [node.name for node in nodes],
        "internals": internals,
        "initial_states": initials,
        "variables_order": list(variables.keys()),
        "variables": variables
    }


def generate_heading():
    return '''#include "types.h"
'''


def generate_aggregate_function(nodes):

    aggregate_function = '''
__device__ float compute_transition_rates(float* __restrict__ transition_rates, const state_t& state)
{
    float sum = 0;
    float tmp;
'''

    for i, node in enumerate(nodes):
        aggregate_function += f'''
    tmp = {node.name}_rate(state);
    transition_rates[{i}] = tmp;
    sum += tmp;
'''

    aggregate_function += '''
    return sum;
}
'''

    return aggregate_function


def is_trivial_logic_function(node, nodes, variables):
    up_expr = node.attributes['rate_up']
    down_expr = node.attributes['rate_down']

    if type(up_expr) == TernExpr and type(up_expr.cond) == Alias and up_expr.cond.name == 'logic' \
            and type(down_expr) == TernExpr and type(down_expr.cond) == Alias and down_expr.cond.name == 'logic' \
            and up_expr.false_branch.evaluate(variables) == 0 and down_expr.true_branch.evaluate(variables) == 0:
        return True


def generate_node_transition_fuction(node, nodes, variables, runtime):

    node_name = node.name

    state_expr = Id(node_name).generate_code(variables, nodes, node, runtime)
    up_expr = node.attributes['rate_up'].generate_code(
        variables, nodes, node, runtime)
    down_expr = node.attributes['rate_down'].generate_code(
        variables, nodes, node, runtime)

    return f'''
__device__ float {node_name}_rate(const state_t& state)
{{
    return {state_expr} ? 
        ({down_expr}) : 
        ({up_expr});
}}
'''


def generate_runtime_variables_code(variables, runtime):
    content = ''

    # generate constant memory array
    if runtime and len(variables.keys()) != 0:
        content += f'''
__constant__ float constant_vars[{len(variables)}];
'''
    content += f'''
cudaError_t set_variables(const float* vars)
{{'''
    if runtime:
        content += f'''
    return cudaMemcpyToSymbol(constant_vars, vars, {len(variables)} * sizeof(float));'''
    else:
        content += f'''
    return cudaSuccess;'''

    content += f'''
}}
'''
    return content


def generate_if_newer(path, content):
    if os.path.exists(path):
        with open(path, 'r') as f:
            old_content = f.read()
    else:
        old_content = ""

    if content != old_content:
        with open(path, 'w') as f:
            f.write(content)


def generate_tr_cu_file(tr_cu_file, nodes, variables, runtime):
    tr_cu_path = 'src/' + tr_cu_file

    content = generate_heading()

    # generate constant memory array and function that sets it
    content += generate_runtime_variables_code(variables, runtime)

    # generate transition functions
    for node in nodes:
        content += generate_node_transition_fuction(
            node, nodes, variables, runtime)

    # generate aggregate function
    content += generate_aggregate_function(nodes)

    generate_if_newer(tr_cu_path, content)


def generate_tr_h_file(tr_h_file, nodes, cfg_program):
    tr_h_path = 'src/' + tr_h_file

    content = generate_tr_header_file(nodes, cfg_program)

    generate_if_newer(tr_h_path, content)


def generate_cfg_file(cfg_file, nodes, cfg_program, variables):
    content = generate_config(nodes, cfg_program, variables)

    with open(cfg_file, 'w') as f:
        f.write(json.dumps(content, indent=4))


def generate_files(bnd_stream, cfg_stream, json_file, runtime_flag):

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

    tr_cu_file = 'transition_rates.cu.generated'
    tr_h_file = 'transition_rates.h.generated'

    generate_tr_cu_file(tr_cu_file, nodes, variables, runtime_flag)
    generate_tr_h_file(tr_h_file, nodes, cfg_program)
    generate_cfg_file(json_file, nodes, cfg_program, variables)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates CUDA code from .bnd and .cfg files and transforms .cfg file to .json file')
    parser.add_argument('bnd_file', type=str, help='input .bnd file')
    parser.add_argument('cfg_file', type=str, help='input .cfg file')
    parser.add_argument('json_file', type=str, help='output .json file')
    parser.add_argument('--runtime', action='store_true', help='generate code such that boolean formulae variables can be changed at runtime')

    args = parser.parse_args()

    with open(args.bnd_file, 'r') as bnd:
        bnd_stream = bnd.read()
    with open(args.cfg_file, 'r') as cfg:
        cfg_stream = cfg.read()

    generate_files(bnd_stream, cfg_stream, args.json_file, args.runtime)
