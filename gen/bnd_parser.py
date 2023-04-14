import ply.yacc as yacc
from bnd_types import *

# Get the token map from the lexer.  This is required.
from bnd_lexer import tokens

def p_program(p):
    '''
    program : node_declaration
            | program node_declaration
    '''
    if (len(p) == 2):
        p[0] = [p[1]]
    else:
        p[1].append(p[2])
        p[0] = p[1]

def p_node_declaration(p):
    '''
    node_declaration : NODE IDENTIFIER '{' node_body '}'
    '''
    attributes = {}
    for attribute in p[4]:
        attributes[attribute.name] = attribute.expr
    p[0] = Node(p[2], attributes)

def p_node_body(p):
    '''
    node_body : node_body node_attribute
              | node_attribute
    '''
    if (len(p) == 2):
        p[0] = [p[1]]
    else:
        p[1].append(p[2])
        p[0] = p[1]

def p_node_attribute(p):
    '''
    node_attribute : IDENTIFIER '=' expression ';'
    '''
    p[0] = Attribute(p[1], p[3])

def p_term_number(p):
    '''
    term : NUMBER
    '''
    p[0] = Lit(int(p[1]))

def p_term_real(p):
    '''
    term : REAL
    '''
    p[0] = Lit(float(p[1]))

def p_term_identifier(p):
    '''
    term : IDENTIFIER
    '''
    p[0] = Id(p[1])

def p_term_variable(p):
    '''
    term : VARIABLE
    '''
    p[0] = Var(p[1])

def p_term_alias(p):
    '''
    term : ALIAS
    '''
    p[0] = Alias(p[1])

def p_term_parens(p):
    '''
    term : '(' expression ')'
    '''
    p[0] = ParExpr(p[2])

def p_unary_expression(p):
    '''
    unary_expression : NOT unary_expression
                     | '+' unary_expression
                     | '-' unary_expression
                     | term
    '''
    if (len(p) == 2):
        p[0] = p[1]
    else:
        p[0] = UnExpr(p[1], p[2])

def p_multiplicative_expression(p):
    '''
    multiplicative_expression : unary_expression
                              | multiplicative_expression '*' unary_expression
                              | multiplicative_expression '/' unary_expression
    '''
    if (len(p) == 2):
        p[0] = p[1]
    else:
        p[0] = BinExpr(p[2], p[1], p[3])


def p_additive_expression(p):
    '''
    additive_expression : multiplicative_expression
                        | additive_expression '+' multiplicative_expression
                        | additive_expression '-' multiplicative_expression
    '''
    if (len(p) == 2):
        p[0] = p[1]
    else:
        p[0] = BinExpr(p[2], p[1], p[3])

def p_relational_expression(p):
    '''
    relational_expression : additive_expression
                          | relational_expression '<' additive_expression
                          | relational_expression '>' additive_expression
                          | relational_expression LE additive_expression
                          | relational_expression GE additive_expression
    '''
    if (len(p) == 2):
        p[0] = p[1]
    else:
        p[0] = BinExpr(p[2], p[1], p[3])

def p_equality_expression(p):
    '''
    equality_expression : relational_expression
                        | equality_expression EQ relational_expression
                        | equality_expression NE relational_expression
    '''
    if (len(p) == 2):
        p[0] = p[1]
    else:
        p[0] = BinExpr(p[2], p[1], p[3])

def p_and_expression(p):
    '''
    and_expression : equality_expression
                   | and_expression AND equality_expression
    '''
    if (len(p) == 2):
        p[0] = p[1]
    else:
        p[0] = BinExpr(p[2], p[1], p[3])

def p_or_expression(p):
    '''
    or_expression : and_expression
                  | or_expression OR and_expression
    '''
    if (len(p) == 2):
        p[0] = p[1]
    else:
        p[0] = BinExpr(p[2], p[1], p[3])

def p_xor_expression(p):
    '''
    xor_expression : or_expression
                   | xor_expression XOR or_expression
    '''
    if (len(p) == 2):
        p[0] = p[1]
    else:
        p[0] = BinExpr(p[2], p[1], p[3])

def p_ternary_expression(p):
    '''
    ternary_expression : xor_expression
                       | xor_expression '?' expression ':' ternary_expression
    '''
    if (len(p) == 2):
        p[0] = p[1]
    else:
        p[0] = TernExpr(p[1], p[3], p[5])

def p_expression(p):
    '''
    expression : ternary_expression
    '''
    p[0] = p[1]

# Error rule for syntax errors
def p_error(p):
    print(f"Syntax error in input! {p}")

# Build the parser
bnd_parser = yacc.yacc()
