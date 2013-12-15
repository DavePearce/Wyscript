package wyscript.par.loop;

import java.util.HashMap;
import wyscript.Interpreter;
import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.ParFor;

public class GPUNestedLoop extends GPULoop{
	private Expr innerExpression;

	public GPUNestedLoop(ParFor loop) {
		super(loop);
		if (loop.getBody().size()>0){
			Stmt first = loop.getBody().get(0);
			if (first instanceof Stmt.ParFor) {
				innerExpression = ((Stmt.ParFor) first).getSource();
			}else{
				throw new IllegalArgumentException("GPUNestedLoop must have " +
						"parallel-for loop as first argument");
			}
		}
		/*gpu nested loop needs to scan indexOf operations and determine
		the inner upper bound*/
	}

	@Override
	public int innerLowerBound(HashMap<String, Object> frame) {
		Interpreter interpreter = new Interpreter();
		return GPULoop.lowerBound(frame, innerExpression, interpreter);
	}

	@Override
	public int innerUpperBound(HashMap<String, Object> frame) {
		Interpreter interpreter = new Interpreter();
		return GPULoop.upperBound(frame, innerExpression, interpreter);
	}

}
