define expr as int | {int op, expr left, expr right}

void main([[char]] args):
    e = {op:1,left:1,right:2}
    println(str(e))
