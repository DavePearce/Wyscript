package wyscript.par.loop;

import java.util.HashMap;

import wyscript.lang.Expr;
import wyscript.lang.Stmt.ParFor;

public abstract class GPUNestedLoop extends GPULoop {

	public GPUNestedLoop(ParFor loop) {
		super(loop);
	}

	@Override
	public abstract int innerLowerBound(HashMap<String, Object> frame);

	@Override
	public abstract int innerUpperBound(HashMap<String, Object> frame);

	public abstract Expr.Variable getInnerIndexVar();

}
