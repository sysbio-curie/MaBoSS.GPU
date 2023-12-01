import numpy as np
import argparse


def generate_bnd(file_name, nodes_count):


    clause_count_max = int(np.sqrt(nodes_count))
    clause_length_max = clause_count_max

    clause_count_max = 5
    clause_length_max = 2

    with open(file_name, "w") as f:
        for i in range(nodes_count - 2):
            f.write(f"Node N{i} {{\n")
            f.write(f"    logic = ")

            clauses_count = np.random.randint(1, clause_count_max + 1)

            clauses = []
            for j in range(clauses_count):
                clause_length = np.random.randint(1, clause_length_max + 1)
                clause = np.random.choice(
                    [f"N{k}" for k in range(nodes_count)], clause_length, replace=False
                )
                # add negation
                for k in range(clause_length):
                    if np.random.rand() < 0.5:
                        clause[k] = f"!{clause[k]}"
                clause = "(" + " & ".join(clause) + ")"
                clauses.append(clause)

            f.write("(" + " | ".join(clauses) + f") & !N{nodes_count - 1}")

            f.write(";\n")
            f.write(f"    rate_up = @logic ? $u_N{i} : 0;\n")
            f.write(f"    rate_down = @logic ? 0 : $d_N{i};\n")

            f.write("}\n\n")
        
        # last-1 node
        f.write(f"Node N{nodes_count - 2} {{\n")
        f.write(f"    logic = !N{nodes_count - 2} & !N{nodes_count - 1};\n")
        f.write(f"    rate_up = @logic ? $u_N{nodes_count - 2} : 0;\n")
        f.write(f"    rate_down = @logic ? 0 : $d_N{nodes_count - 2};\n")
        f.write("}\n\n")
        
        # last node
        f.write(f"Node N{nodes_count - 1} {{\n")
        f.write(f"    logic = !N{nodes_count - 1};\n")
        f.write(f"    rate_up = @logic ? $u_N{nodes_count - 1} : 0;\n")
        f.write(f"    rate_down = @logic ? 0 : $d_N{nodes_count - 1};\n")
        f.write("}\n\n")



def generate_cfg(
    file_name,
    nodes_count,
    noninternals_count,
    sample_count,
    time_tick,
    max_time,
    discrete_time,
):
    with open(file_name, "w") as f:
        f.write(f"sample_count = {sample_count};\n")
        f.write(f"time_tick = {time_tick};\n")
        f.write(f"max_time = {max_time};\n")
        f.write(f"discrete_time = {discrete_time};\n")

        for i in range(nodes_count - 1):
            f.write(f"$u_N{i} = 1;\n")
            f.write(f"$d_N{i} = 1;\n")
            f.write(f"N{i}.istate = 0;\n")
            if i < noninternals_count:
                f.write(f"N{i}.is_internal = 0;\n")
            else:
                f.write(f"N{i}.is_internal = 1;\n")

        f.write(f"$u_N{nodes_count - 1} = 0.01;\n")
        f.write(f"$d_N{nodes_count - 1} = 0;\n")
        f.write(f"N{nodes_count - 1}.istate = 0;\n")
        f.write(f"N{nodes_count - 1}.is_internal = 1;\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_prefix")
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--noninternals", type=int, default=5)
    parser.add_argument("--sample_count", type=int, default=1000000)
    parser.add_argument("--time_tick", type=int, default=5)
    parser.add_argument("--max_time", type=int, default=100)
    parser.add_argument("--discrete", action="store_true")
    args = parser.parse_args()

    generate_bnd(args.file_prefix + ".bnd", args.nodes)
    generate_cfg(
        args.file_prefix + ".cfg",
        args.nodes,
        args.noninternals,
        args.sample_count,
        args.time_tick,
        args.max_time,
        args.discrete,
    )
