define Expr as real | [Expr]
define Value as real | [Value]

Value init():
    return 0.0123

void main([[char]] args):
    v = init()
    if v ~= [Expr]:
        println("GOT LIST")
    else:
        println(str(v))
