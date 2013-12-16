package wyscript.par.loop;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import wyscript.Interpreter;
import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.lang.Expr.BOp;
import wyscript.lang.Expr.Binary;
import wyscript.par.util.Argument;
import wyscript.util.SyntaxError.InternalFailure;

public abstract class GPULoop {
	private List<Argument> arguments;
	private HashMap<Argument,String> nameMap = new HashMap<Argument,String>();

	public GPULoop(Stmt.ParFor loop) {
		this.loop = loop;
	}
	/**
	 * Initialises the argument list of this GPULoop, allowing them to be used
	 * for writing
	 * @param env
	 * @return
	 */
	public List<Argument> initialiseArguments(HashMap<String,Type> env) {
		List<String> parameters = new ArrayList<String>();
		Set<String> nonparameters = new HashSet<String>();
		//now ensure there are no name clashes
		List<Argument> arguments = GPUUtils.scanForFunctionParameters(loop.getBody(), parameters, nonparameters, env);
		//now mangle name
		for (Argument arg : arguments) {
			int suffix = 0;
			String mangledName = arg.name;
			if (arg instanceof Argument.Length1D || arg instanceof Argument.Length2D) {
				while (parameters.contains(mangledName)) {
					String identifier = "MANGLED";
					mangledName = arg.name + identifier  + Integer.toString(suffix++);
				}
				nameMap.put(arg, mangledName);
			}else {
				nameMap.put(arg, arg.name);
			}
		}
		return arguments;
	}
	public List<Argument> getArguments() {
		return arguments;
	}
	public boolean isArgument(String name) {
		for (Argument arg : arguments) {
			if (arg.name.equals(name)) return true;
		}
		return false;
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
	public Expr.Variable getIndexVar() {
		return loop.getIndex();
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
			Expr rhs = binary.getRhs();
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
	public String kernelName(Argument arg) {
		return nameMap.get(arg.name);
	}
}