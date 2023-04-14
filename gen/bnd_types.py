from typing import NamedTuple


def get_node_idx(node_name, nodes):
    for i, node in enumerate(nodes):
        if node.name == node_name:
            return i
    raise ValueError('Unknown node name: ' + node_name)


class Node(NamedTuple):
    name: str
    attributes: dict


class Attribute(NamedTuple):
    name: str
    expr: object


class BinExpr(NamedTuple):
    op: str
    left: object
    right: object

    def evaluate(self, variables):
        if self.op == '+':
            return self.left.evaluate(variables) + self.right.evaluate(variables)
        elif self.op == '-':
            return self.left.evaluate(variables) - self.right.evaluate(variables)
        elif self.op == '*':
            return self.left.evaluate(variables) * self.right.evaluate(variables)
        elif self.op == '/':
            return self.left.evaluate(variables) / self.right.evaluate(variables)
        elif self.op == '==':
            return self.left.evaluate(variables) == self.right.evaluate(variables)
        elif self.op == '!=':
            return self.left.evaluate(variables) != self.right.evaluate(variables)
        elif self.op == '<':
            return self.left.evaluate(variables) < self.right.evaluate(variables)
        elif self.op == '<=':
            return self.left.evaluate(variables) <= self.right.evaluate(variables)
        elif self.op == '>':
            return self.left.evaluate(variables) > self.right.evaluate(variables)
        elif self.op == '>=':
            return self.left.evaluate(variables) >= self.right.evaluate(variables)
        else:
            raise ValueError('Unknown binary operator: ' + self.op)

    def generate_code(self, variables, nodes, curr_node):
        mod_op = self.op
        if self.op.lower() == 'and' or self.op == '&':
            mod_op = '&&'
        elif self.op.lower() == 'or' or self.op == '|':
            mod_op = '||'
        return f"{self.left.generate_code(variables, nodes, curr_node)} {mod_op} {self.right.generate_code(variables, nodes, curr_node)}"


class UnExpr(NamedTuple):
    op: str
    expr: object

    def evaluate(self, variables):
        if self.op == '-':
            return -self.expr.evaluate(variables)
        elif self.op == '+':
            return self.expr.evaluate(variables)
        elif self.op == '!' or self.op.lower() == 'not':
            return not self.expr.evaluate(variables)
        else:
            raise ValueError('Unknown unary operator: ' + self.op)

    def generate_code(self, variables, nodes, curr_node):
        return f"{self.op}{self.expr.generate_code(variables, nodes, curr_node)}"


class TernExpr(NamedTuple):
    cond: object
    true_branch: object
    false_branch: object

    def evaluate(self, variables):
        if self.cond.evaluate(variables):
            return self.true_branch.evaluate(variables)
        else:
            return self.false_branch.evaluate(variables)

    def generate_code(self, variables, nodes, curr_node):
        return f"{self.cond.generate_code(variables, nodes, curr_node)} ? {self.true_branch.generate_code(variables, nodes, curr_node)} : {self.false_branch.generate_code(variables, nodes, curr_node)}"


class ParExpr(NamedTuple):
    expr: object

    def evaluate(self, variables):
        return self.expr.evaluate(variables)

    def generate_code(self, variables, nodes, curr_node):
        return f"({self.expr.generate_code(variables, nodes, curr_node)})"


class Id(NamedTuple):
    name: str

    def generate_code(self, variables, nodes, curr_node):
        idx = get_node_idx(self.name, nodes)
        return f"(state & {1 << idx})"


class Var(NamedTuple):
    name: str

    def evaluate(self, variables):
        return variables[self.name]

    def generate_code(self, variables, nodes, curr_node):
        return str(variables[self.name])


class Alias(NamedTuple):
    name: str

    def generate_code(self, variables, nodes, curr_node):
        return curr_node.attributes[self.name].generate_code(variables, nodes, curr_node)


class Lit(NamedTuple):
    value: object

    def evaluate(self, variables):
        return self.value

    def generate_code(self, variables, nodes, curr_node):
        return str(self.value)
