#!/usr/bin/python3

import argparse, subprocess, sys, shutil, os

JSON_FILE_PATH = "/tmp/mabossg.json"
BUILD_PATH = "/tmp/MaBoSSG-build"

def generate_model(bnd_file, cfg_file, runtime_variables=False, runtime_internals=False):
    
    res_gen = subprocess.Popen("python3 gen/generator.py %s%s%s %s %s" % (
        "" if not runtime_variables else "--runtime_variables ",
        "" if not runtime_internals else "--runtime_internals ",
        bnd_file, cfg_file, 
        JSON_FILE_PATH
    ), stdout=subprocess.PIPE, shell=True)
    out, err = res_gen.communicate()
    # print(out)
    if err is not None and len(err) > 0:
        print(err, file=sys.stderr)
    
    if res_gen.returncode != 0:
        exit(res_gen.returncode)


def compile_model():    
    
    if os.path.exists("build"):
        shutil.rmtree("build")

    res_comp = subprocess.Popen("cmake -Wno-dev -DCMAKE_BUILD_TYPE=Release -B %s .; cmake --build %s" % (
        BUILD_PATH, BUILD_PATH
        ), stdout=subprocess.PIPE, shell=True
    )
    
    out, err = res_comp.communicate()
    # print(out)
    if err is not None and len(err) > 0:
        print(err, file=sys.stderr)

    if res_comp.returncode != 0:
        exit(res_comp.returncode)

        
def run_model(res_prefix):
        
    res_run = subprocess.Popen("%s/MaBoSSG -o %s %s" % (BUILD_PATH, res_prefix, JSON_FILE_PATH), stdout=subprocess.PIPE, shell=True)
    out, err = res_run.communicate()
    # print(out)
    if err is not None and len(err) > 0:
        print(err, file=sys.stderr)
    
    if res_run.returncode != 0:
        exit(res_run.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate MaBoSS model using MaBoSSG"
    )
    parser.add_argument("bnd_file", type=str, help="input .bnd file")
    parser.add_argument("-c", "--cfg_file", type=str, help="input .cfg file", required=True)
    parser.add_argument("-o", "--res_prefix", type=str, required=True)
    parser.add_argument(
        "--runtime-variables",
        action="store_true",
        help="generate code such that boolean formulae variables can be changed at runtime",
    )
    parser.add_argument(
        "--runtime-internals",
        action="store_true",
        help="generate code such that internal nodes can be changed at runtime",
    )
    args = parser.parse_args()

    generate_model(args.bnd_file, args.cfg_file, args.runtime_variables, args.runtime_internals)
    compile_model()
    run_model(args.res_prefix)
