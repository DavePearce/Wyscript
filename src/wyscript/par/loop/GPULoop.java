package wyscript.par.loop;

import java.util.HashMap;
import java.util.List;

import wyscript.Interpreter;
import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Expr.BOp;
import wyscript.lang.Expr.Binary;
import wyscript.par.util.Argument;
import wyscript.util.SyntaxError.InternalFailure;

public abstract class GPULoop {
	public GPULoop(Stmt.ParFor loop) {
		this.loop = loop;
	}
	protected final Stmt.ParFor loop;
	public abstract Stmt.ParFor getLoop();
	public abstract List<Argument> getArguments();
	/**
	 * Returns the lower index bound for the outer loop.
	 * @param frame
	 * @return
	 */
	public int outerLowerBound(HashMap<String,Object> frame){
		Expr src = loop.getSource();
		Interpreter interpreter = new Interpreter();
		if (src instanceof Expr.Binary) {
			Expr.Binary binary = (Binary) src;
			Expr lhs = binary.getLhs();
			if (binary.getOp()==BOp.RANGE) {
				return (Integer) interpreter.execute(lhs,frame);
			}else {
				InternalFailure.internalFailure("Could not compute lower bound for range ", "FILE_UNKNOWN", loop);
			}
		}else if (src instanceof Expr.Variable) {
			return 0;
		}else {
			InternalFailure.internalFailure("Could not compute lower bound for range ", "FILE_UNKNOWN", loop);
		}
		return -1;
	}
	/**
	 * Returns the higher index bound for the outer loop.
	 * @param frame
	 * @return
	 */
	public int outerUpperBound(HashMap<String,Object> frame){
		Expr src = loop.getSource();
		Interpreter interpreter = new Interpreter();
		if (src instanceof Expr.Binary) {
			Expr.Binary binary = (Binary) src;
			Expr rhs = binary.getRhs();
			if (binary.getOp()==BOp.RANGE) {
				return (Integer) interpreter.execute(rhs,frame);
			}else {
				InternalFailure.internalFailure("Could not compute lower bound for range ", "FILE_UNKNOWN", loop);
			}
		}
		return 0;
	}
	/**
	 * Returns -1 if this loop is flat
	 * @param frame
	 * @return
	 */
	public abstract int innerLowerBound(HashMap<String,Object> frame);
	/**
	 * Returns -1 if this loop is flat
	 * @param frame
	 * @return
	 */
	public abstract int innerUpperBound(HashMap<String,Object> frame);
}