package wyscript.par.loop;

import java.util.HashMap;

import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.BoundCalc;
import wyscript.lang.Stmt.ParFor;
import wyscript.util.SyntaxError.InternalFailure;

public abstract class GPUNestedLoop extends GPULoop {

	public GPUNestedLoop(ParFor loop) {
		super(loop);
	}

	@Override
	public int innerLowerBound(HashMap<String, Object> frame) {
		return boundCalc.getLowY();
	}

	@Override
	public int innerUpperBound(HashMap<String, Object> frame) {
		return boundCalc.getHighY();
	}

	public abstract Expr.Variable getInnerIndexVar();

}
