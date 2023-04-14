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
