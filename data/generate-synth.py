import numpy as np
import argparse


def generate_bnd(file_name, nodes_count, stripe_count, make_fixed_node):
    node_strings = []

    if make_fixed_node:
        nodes_count -= 1

    for i in range(nodes_count):
        node_str = f"Node N{i} {{\n"
        node_str += f"    logic = "

        up_nodes = []
        down_nodes = []

        for j in range(1, stripe_count + 1):
            up_nodes.append(f"N{(i - j + nodes_count) % nodes_count}")
            down_nodes.append(f"N{(i + j) % nodes_count}")
        up_nodes.append(f"!N{(i - stripe_count - 1 + nodes_count) % nodes_count}")

        node_str += f"(!N{i} & "

        if make_fixed_node:
            node_str += " !NFP & "

        node_str += (
            " & ".join(up_nodes)
            + f") | (!(N{i} & "
            + " & ".join(down_nodes)
            + f") & N{i}"
        )

        if make_fixed_node:
            node_str += " & !NFP"

        node_str += ")"

        if make_fixed_node:
            node_str += f" | (NFP)"

        node_str += ";\n"
        node_str += f"    rate_up = @logic ? $u_N{i} : 0;\n"
        node_str += f"    rate_down = @logic ? 0 : $d_N{i};\n"

        node_str += "}\n\n"

        node_strings.append(node_str)

    # shuffle nodes
    # np.random.shuffle(node_strings)

    if make_fixed_node:
        node_str = f"Node NFP {{\n"
        node_str += f"    logic = !NFP;\n"
        node_str += f"    rate_up = @logic ? $u_NFP : 0;\n"
        node_str += f"    rate_down = @logic ? 0 : $d_NFP;\n"
        node_str += "}\n\n"

        node_strings.append(node_str)

    with open(file_name, "w") as f:
        f.write("".join(node_strings))


def generate_cfg(
    file_name,
    nodes_count,
    noninternals_count,
    sample_count,
    time_tick,
    max_time,
    discrete_time,
    stripe_count,
    internal_stride,
    fixed_node_prob,
):
    with open(file_name, "w") as f:
        f.write(f"sample_count = {sample_count};\n")
        f.write(f"time_tick = {time_tick};\n")
        f.write(f"max_time = {max_time};\n")
        f.write(f"discrete_time = {discrete_time};\n")

        if fixed_node_prob != 0:
            nodes_count -= 1
            f.write(f"$u_NFP = {fixed_node_prob};\n")
            f.write(f"$d_NFP = 0;\n")
            f.write(f"NFP.istate = 0;\n")
            f.write(f"NFP.is_internal = 0;\n")

        for i in range(nodes_count):
            f.write(f"$u_N{i} = 1;\n")
            f.write(f"$d_N{i} = 1;\n")
            f.write(f"N{i}.istate = {1 if i < stripe_count else 0};\n")
            if i % internal_stride == 0 and noninternals_count > 0:
                f.write(f"N{i}.is_internal = 0;\n")
                noninternals_count -= 1
            else:
                f.write(f"N{i}.is_internal = 1;\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic data generator. The nodes in this boolean model form a ring. Throughout the ring, a signal of specific length is being propagated. The model has also one special node, which is able to push the simulation state to a single fixed point. The model is highly customizable - one can specify the size of the ring or the lenght of nodes' formulas by modifying the signal length."
    )
    parser.add_argument("file_prefix")
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--noninternals", type=int, default=5)
    parser.add_argument("--sample_count", type=int, default=1000000)
    parser.add_argument("--time_tick", type=int, default=5)
    parser.add_argument("--max_time", type=int, default=100)
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument(
        "--signal_length",
        type=int,
        default=0,
        help="The size of signal propagated throughout the graph which specifies the size of nodes' formulas in the form of 2 * {signal_length} + 2. If not assigned, it will be set to int(sqrt(nodes))",
    )
    parser.add_argument(
        "--continous_noninternals",
        action="store_true",
        help="Noninternals are right next to each other creating bigger pressure on windows computation",
    )
    parser.add_argument(
        "--steps_to_fixed_point",
        type=int,
        default=0,
        help="The average number of steps after which the trajectory arrives to a fixed point",
    )
    args = parser.parse_args()

    stripe_count = (
        args.signal_length if args.signal_length > 0 else int(np.sqrt(args.nodes))
    )
    internal_stride = (
        1 if args.continous_noninternals else args.nodes // args.noninternals
    )

    generate_bnd(
        args.file_prefix + ".bnd",
        args.nodes,
        stripe_count,
        args.steps_to_fixed_point != 0,
    )
    generate_cfg(
        args.file_prefix + ".cfg",
        args.nodes,
        args.noninternals,
        args.sample_count,
        args.time_tick,
        args.max_time,
        args.discrete,
        stripe_count,
        internal_stride,
        1 / args.steps_to_fixed_point if args.steps_to_fixed_point != 0 else 0,
    )
