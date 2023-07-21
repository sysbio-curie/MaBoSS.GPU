from typing import NamedTuple

class ConstantDeclaration(NamedTuple):
    name: str
    expr: object

    def evaluate(self, variables):
        return self.expr.evaluate(variables)

class VarDeclaration(NamedTuple):
    name: str
    expr: object

    def evaluate(self, variables):
        return self.expr.evaluate(variables)

class AttrDeclaration(NamedTuple):
    name: str
    attr: str
    expr: object

    def evaluate(self, variables):
        return self.expr.evaluate(variables)

class IstateDeclaration(NamedTuple):
    name: str
    prob1: object
    prob2: object


class IstateProbability(NamedTuple):
    value_prob: object
    value: int

    def evaluate(self, variables):
        return self.value_prob.evaluate(variables)
