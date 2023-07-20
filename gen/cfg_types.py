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
    decl1: object
    decl2: object
    
class IstateProbability(NamedTuple):
    expr: object
    value: int
    
    def evaluate(self, variables):
        return self.expr.evaluate(variables)
