package wyscript.par.loop;

import java.util.HashMap;
import wyscript.Interpreter;
import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Expr.BOp;
import wyscript.lang.Expr.Binary;
import wyscript.util.SyntaxError.InternalFailure;

public abstract class GPULoop {
	public GPULoop(Stmt.ParFor loop) {
		this.loop = loop;
	}
	protected final Stmt.ParFor loop;

	public Stmt.ParFor getLoop() {
		return this.loop;
	}
	/**
	 * Returns the lower index bound for the outer loop.
	 * @param frame
	 * @return
	 */
	public int outerLowerBound(HashMap<String,Object> frame){
		Expr src = loop.getSource();
		Interpreter interpreter = new Interpreter();
		return lowerBound(frame, src, interpreter);
	}
	/**
	 * Returns the higher index bound for the outer loop.
	 * @param frame
	 * @return
	 */
	public int outerUpperBound(HashMap<String,Object> frame){
		Expr src = loop.getSource();
		Interpreter interpreter = new Interpreter();
		return upperBound(frame, src, interpreter);
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

	public static int upperBound(HashMap<String, Object> frame, Expr src,
			Interpreter interpreter) {
		if (src instanceof Expr.Binary) {
			Expr.Binary binary = (Binary) src;
			Expr rhs = binary.getLhs();
			if (binary.getOp()==BOp.RANGE) {
				return (Integer) interpreter.execute(rhs,frame);
			}else {
				InternalFailure.internalFailure("Could not compute upper bound for range ", "FILE_UNKNOWN", src);
			}
		}
		return 0;
	}
	public static int lowerBound(HashMap<String, Object> frame, Expr src,
			Interpreter interpreter) {
		if (src instanceof Expr.Binary) {
			Expr.Binary binary = (Binary) src;
			Expr lhs = binary.getLhs();
			if (binary.getOp()==BOp.RANGE) {
				return (Integer) interpreter.execute(lhs,frame);
			}else {
				InternalFailure.internalFailure("Could not compute lower bound for range ", "FILE_UNKNOWN", src);
			}
		}
		return 0;
	}
	public String kernelName(String name) {
		return name; //TODO make this work properly
	}
	public String lengthName(String name) {
		return name + "_length";

	}
	public String widthName(String name) {
		return name + "_width";

	}
	public String heightName(String name) {
		return name + "_height";

	}
}