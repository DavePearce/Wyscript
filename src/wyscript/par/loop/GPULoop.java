package wyscript.par.loop;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;

import wyscript.Interpreter;
import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.lang.Expr.BOp;
import wyscript.lang.Expr.Binary;
import wyscript.lang.Stmt.BoundCalc;
import wyscript.par.util.Argument;
import wyscript.util.SyntaxError.InternalFailure;
/**
 * This abstract class is the parent representation of all parallel
 * loops. It encapsulates the arguments and indices and bridges the
 * namespace of the kernel and the loop.
 * @author Mate Antunovic
 *
 */
public class GPULoop {
	private List<Argument> arguments;
	private HashMap<Argument,String> nameMap = new HashMap<Argument,String>();
	protected final Stmt.ParFor loop;
	private Expr source;

	public GPULoop(Stmt.ParFor loop) {
		this.loop = loop;
	}
	/**
	 * Initialises the argument list of this GPULoop, allowing them to be used
	 * for writing
	 * @param env The type environment at the point of the Wyscript program that parFor loop resides
	 * @return
	 */
	public List<Argument> initialiseArguments(HashMap<String,Type> env) {
		List<String> parameters = new ArrayList<String>();
		Set<String> nonparameters = new HashSet<String>();
		//now ensure there are no name clashes
		if (!env.containsKey("i")) nonparameters.add("i");
		if (!env.containsKey("j")) nonparameters.add("j");
		List<Argument> arguments = GPUUtils.scanForFunctionParameters(loop, parameters, nonparameters, env);
		//now mangle name
		for (Argument arg : arguments) {
			int suffix = 0;
			String mangledName = arg.name;
			if (arg instanceof Argument.Length1D) {
				mangledName = arg.name+"_length";
				while (parameters.contains(mangledName)) {
					String identifier = "_length";
					mangledName = arg.name + identifier  + Integer.toString(suffix++);
				}
				nameMap.put(arg, mangledName);
			}else if (arg instanceof Argument.Length2D) {
				String identifier;
				if (((Argument.Length2D) arg).isHeight) {
					identifier = "_height";
				}else {
					identifier = "_width";
				}
				mangledName = arg.name + identifier;
				while (parameters.contains(mangledName)) {
					mangledName = arg.name + identifier  + Integer.toString(suffix++);
				}
				nameMap.put(arg, mangledName);
			}else {
				nameMap.put(arg, arg.name);
			}
		}
		this.arguments = arguments;
		return arguments;
	}
	public List<Argument> getArguments() {
		return arguments;
	}
	/**
	 * @param name
	 * @return True if at least one argument has the name
	 */
	public boolean isArgument(String name) {
		for (Argument arg : arguments) {
			if (arg.name.equals(name)) return true;
		}
		return false;
	}

	public Stmt.ParFor getLoop() {
		return this.loop;
	}
	/**
	 * Returns the lower index bound for the outer loop.
	 * @param frame
	 * @return
	 */
	public int outerLowerBound(HashMap<String,Object> frame){
		return loop.getCalc().getLowX();
	}
	/**
	 * Returns the higher index bound for the outer loop.
	 * @param frame
	 * @return
	 */
	public int outerUpperBound(HashMap<String,Object> frame){
		return loop.getCalc().getHighX();
	}
	/**
	 * Returns -1 if this loop is flat
	 * @param frame
	 * @return
	 */
	public int innerLowerBound(HashMap<String,Object> frame){
		return loop.getCalc().getLowY();
		}
	/**
	 * Returns -1 if this loop is flat
	 * @param frame
	 * @return
	 */
	public int innerUpperBound(HashMap<String,Object> frame){
		return loop.getCalc().getHighY();
		}
	//TODO Delete me
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
	//TODO Delete me
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
	/**
	 * Returns the name of this argument within the kernel
	 * (i.e. it can be accessed from within the kernel code)
	 * @param arg
	 * @return
	 */
	public String kernelName(Argument arg) {
		String result = nameMap.get(arg);
		if (result == null) return arg.name;
		else return result;
	}
	/**
	 * Converts a name within the Wyscript program to its form wihtin the kernel
	 * @param name
	 * @return
	 */
	public String kernelName(String name) {
		for (Argument arg : arguments) {
			if (name.equals(arg.name)) {
				return nameMap.get(arg);
			}
		}
		throw new NoSuchElementException("Could not find variable for "+name);
	}
	public String lengthName(String name) {
		for (Argument arg : arguments) {
			if (name.equals(arg.name)) {
				if (arg instanceof Argument.Length1D) return nameMap.get(arg);
			}
		}
		throw new NoSuchElementException("Could not find length variable for "+name);
	}
	public String widthName(String name) {
		for (Argument arg : arguments) {
			if (name.equals(arg.name) && arg instanceof Argument.Length2D) {
				 if (!((Argument.Length2D)arg).isHeight) return nameMap.get(arg);
			}
		}
		throw new NoSuchElementException("Could not find width variable for "+name);
	}
	/**
	 * Takes a matrix name and returns the name of its height variable
	 * @param name
	 * @return
	 */
	public String heightName(String name) {
		for (Argument arg : arguments) {
			if (name.equals(arg.name) && arg instanceof Argument.Length2D) {
				 if (((Argument.Length2D)arg).isHeight) return nameMap.get(arg);
			}
		}
		throw new NoSuchElementException("Could not find height variable for "+name);
	}
	/**
	 *
	 * @return
	 */
	public Expr source() { //TODO Complete me
		return source;
	}
}