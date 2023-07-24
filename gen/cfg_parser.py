import ply.yacc as yacc
from bnd_types import *
from cfg_types import *

# Get the token map from the lexer.  This is required.
from cfg_lexer import tokens

def p_program(p):
    '''
    program : declaration
            | program declaration
    '''
    if (len(p) == 2):
        p[0] = [p[1]]
    else:
        p[1].append(p[2])
        p[0] = p[1]

def p_declaration(p):
    '''
    declaration : attr_declaration
                | var_declaration
                | const_declaration
                | istate_declaration
    '''
    p[0] = p[1]

def p_var_declaration(p):
    '''
    var_declaration : VARIABLE '=' expression ';'
    '''
    p[0] = VarDeclaration(p[1], p[3])

def p_attr_declaration(p):
    '''
    attr_declaration : IDENTIFIER '.' IDENTIFIER '=' expression ';'
    '''
    p[0] = AttrDeclaration(p[1], p[3], p[5])

def p_const_declaration(p):
    '''
    const_declaration : IDENTIFIER '=' expression ';'
    '''
    p[0] = ConstantDeclaration(p[1], p[3])

def p_istate_declaration(p):
    '''
    istate_declaration : '[' IDENTIFIER ']' '.' IDENTIFIER '=' expression '[' NUMBER ']' ',' expression '[' NUMBER ']' ';'
    '''
    p[0] = IstateDeclaration(p[2], IstateProbability(p[7], p[9]), IstateProbability(p[12], p[14]))

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

def p_term_boolean(p):
    '''
    term : TRUE
         | FALSE
    '''
    p[0] = Lit(p[1].lower() == 'true')

def p_term_variable(p):
    '''
    term : VARIABLE
    '''
    p[0] = Var(p[1])

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
        p[0] = TernExpr(p[1], p[2], p[3])

def p_expression(p):
    '''
    expression : ternary_expression
    '''
    p[0] = p[1]

# Error rule for syntax errors
def p_error(p):
    print(f"Syntax error in input! {p}")

# Build the parser
cfg_parser = yacc.yacc()
