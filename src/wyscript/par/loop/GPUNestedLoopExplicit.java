package wyscript.par.loop;

import java.util.HashMap;

import wyscript.Interpreter;
import wyscript.lang.Expr;
import wyscript.lang.Expr.Variable;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.ParFor;

public class GPUNestedLoopExplicit extends GPUNestedLoop{
	private Expr innerExpression;
	private Variable innerIndex;

	public GPUNestedLoopExplicit(ParFor loop) {
		super(loop);
		for (Stmt stmt : loop.getBody()) {
			if (stmt instanceof Stmt.ParFor) {
				innerExpression = ((Stmt.ParFor) stmt).getSource();
				innerIndex = ((Stmt.ParFor) stmt).getIndex();
			}
			return;
		}
		throw new IllegalArgumentException("GPUNestedLoopExplicit must have " +
				"parallel-for loop as within body");
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

	@Override
	public Variable getInnerIndexVar() {
		return innerIndex;
	}

}
