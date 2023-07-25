import ply.lex as lex

tokens = (
    'AND',
    'OR',
    'XOR',
    'NOT',
    'LE',
    'GE',
    'EQ',
    'NE',
    'IDENTIFIER',
    'NUMBER',
    'REAL',
    'VARIABLE',
    'TRUE',
    'FALSE'
)

literals = ['+', '-', '*', '/', '<', '>', '?',
            ':', '}', '{', ';', '=', '(', ')', '.',
            '[', ']', ',']

t_AND = r'(&)|(&&)'
t_OR = r'(\|)|(\|\|)'
t_XOR = r'\^'
t_NOT = r'!'
t_LE = r'<='
t_GE = r'>='
t_EQ = r'=='
t_NE = r'!='


def t_COMMENT(t):
    r'//.*'
    pass


def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    if t.value.lower() == 'node':
        t.type = 'NODE'
    elif t.value.lower() == 'and':
        t.type = 'AND'
    elif t.value.lower() == 'or':
        t.type = 'OR'
    elif t.value.lower() == 'xor':
        t.type = 'XOR'
    elif t.value.lower() == 'not':
        t.type = 'NOT'
    elif t.value.lower() == 'true':
        t.type = 'TRUE'
    elif t.value.lower() == 'false':
        t.type = 'FALSE'
    return t


def t_REAL(t):
    r'([0-9]+\.[0-9]+)|(\.[0-9]+)|([0-9]+\.)|([0-9]\.?[0-9]*[eE][-+]?[0-9]+)'
    t.value = float(t.value)
    return t


def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t


def t_VARIABLE(t):
    r'\$[a-zA-Z_][a-zA-Z_0-9]*'
    t.value = t.value[1:]
    return t


# Define a rule so we can track line numbers
def t_newline(t):
    r'\r?\n'
    t.lexer.lineno += 1


# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t'

# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


# Build the lexer
cfg_lexer = lex.lex()
