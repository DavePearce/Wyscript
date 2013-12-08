package wyscript.par;

//TODO The entire environment cannot be used from a function

import static wyscript.util.SyntaxError.syntaxError;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.lang.WyscriptFile;
import wyscript.lang.WyscriptFile.FunDecl;
import wyscript.util.SyntaxError.InternalFailure;
import wyscript.util.TypeChecker;

/**
 * The kernel generator is responsible for the generation Cuda kernels
 * from parallel-for loops. It maps each parallel for loop to a KernelRunner.
 * This runner is then attached to the parallel-for loop.
 * @author Mate Antunovic
 *
 */
public class KernelGenerator {
	/**
	 * Scan the Wyscript file and convert available loops to parallel loops
	 * @param wyFile
	 */
	private KernelGenerator(){} //cannot instantiate
	/**
	 * Modifies the AST so that each ParFor loop is connected to its kernel
	 * @param wyFile
	 * @param file
	 * @param environment
	 */
	public static void generateKernels(WyscriptFile wyFile) {
		Map<String , WyscriptFile.FunDecl> functions = new HashMap<String ,
				WyscriptFile.FunDecl>();
		for (WyscriptFile.Decl declaration : wyFile.declarations) {
			if(declaration instanceof WyscriptFile.FunDecl) {
				WyscriptFile.FunDecl fd = (WyscriptFile.FunDecl) declaration;
				functions.put(fd.name(), fd);
			}
		}
		generateKernels(functions,wyFile);
	}
	/**
	 * Scan an individual function for loops that can be converted
	 * @param functions
	 * @param file
	 * @param environment
	 */
	private static void generateKernels(Map<String, FunDecl> functions, WyscriptFile file) {
		for (String fname : functions.keySet()) {
			WyscriptFile.FunDecl func = functions.get(fname);
			//pass the name of the function down so it can be used to address kernel
			scanFuncBody(func , fname , file);
		}
	}
	/**
	 * Scans the function body and identifies parallel for loops
	 * @param function
	 * @param funcname
	 * @param environment
	 */
	private static void scanFuncBody(WyscriptFile.FunDecl function , String funcname , WyscriptFile file) {
		int loopPosition = 0;
		//Map<String, Type> environment = new HashMap<String,Type>();
		//checker.check(function.statements , environment);
		Map<String,Type> env = new HashMap<String,Type>();
		for (int i= 0; i < function.statements.size() ; i++) {
			Stmt statement = function.statements.get(i);
			//update the environment
			env = getEnvironment(statement, file , env);
			if (statement instanceof Stmt.ParFor) {
				Stmt.ParFor loop = (Stmt.ParFor) statement;
				String indexName = loop.getIndex().getName();
				HashMap<String,Type> newEnv = new HashMap<String,Type>(env);
//				newEnv.put(indexName, new Type.Int());
				String id = funcname + Integer.toString(loopPosition);
				KernelRunner runner = generateForKernel((Stmt.ParFor)statement
						, newEnv , id);
				//now plug the kernel runner into the loop
				loop.setKernelRunner(runner);
				loopPosition++;
			}
		}
	}
	/**
	 * Returns a KernelRunner for this loop.
	 * @param loop The loop to be converted for running on GPU
	 * @param environment The type environment so far
	 * @param id The file name under which the loop is invoked
	 * @return
	 */
	private static KernelRunner generateForKernel(Stmt.ParFor loop ,
			Map<String,Type> environment , String id) {
		KernelWriter writer = null;
		writer = new KernelWriter(id, environment, loop);
		return writer.getRunner();
	}
	/**
	 * Returns the entire environment of the function
	 * @param function
	 * @param file
	 * @return
	 */
	public static Map<String,Type> getEnvironment(Stmt stmt, WyscriptFile file,
			Map<String,Type> env) {
		if (stmt instanceof Stmt.VariableDeclaration) {
				getEnvironment((Stmt.VariableDeclaration)stmt,env,file);
		}
		return env;
	}

	private static void getEnvironment(Stmt.VariableDeclaration stmt, Map<String, Type>
	environment, WyscriptFile file) {
		Expr expression = stmt.getExpr();
		String name = stmt.getName();
		environment.put(name, getType(expression,file));
	}
	private static Type getType(Expr expression,WyscriptFile file) {
		if (expression instanceof Expr.Binary) {
			//return the greater of either types
			return getType((Expr.Binary)expression,file);
		}else if (expression instanceof Expr.Constant) {
			Expr.Constant constant = (Expr.Constant)expression;
			return getType(constant,file);
		}else if (expression instanceof Expr.IndexOf) {
			//return the element type of the list
			return getType((Expr.IndexOf)expression,file);
		}else if (expression instanceof Expr.ListConstructor) {
			//return the element of greatest type from the list
			//TODO this may be incorrect.
			return getType((Expr.ListConstructor)expression,file);
		}else if (expression instanceof Expr.Variable) {
			//returns the type of a declared variable
			return getType((Expr.Variable)expression,file);
		}else if (expression instanceof Expr.Cast) {
			return getType((Expr.Cast)expression,file);
		}else if (expression instanceof Expr.Invoke){
			return getType((Expr.Invoke)expression,file);
		}else if (expression instanceof Expr.RecordAccess){
			return getType((Expr.RecordAccess)expression,file);
		}else if (expression instanceof Expr.Unary) {
			Expr.Unary unary = (Expr.Unary)expression;
			return getType(unary.getExpr(),file);
		}
		return null;
	}
	/**
	 * Returns the super of the types in the binary expression (fails
	 * for disjoint types).
	 * @param expression
	 * @param file
	 * @return
	 */
	private static Type getType(Expr.Binary expression, WyscriptFile file) {
		Expr lhs = expression.getLhs();
		Expr rhs = expression.getRhs();
		Type t1 = getType(lhs, file);
		Type t2 = getType(rhs, file);
		return greaterType(t1,t2,file);
	}
	private static Type getType(Expr.Constant constant,WyscriptFile file) {
		Object val = constant.getValue();
		if (val instanceof Integer) return new Type.Int();
		else if (val instanceof Double) return new Type.Real(); //ahem is this correct
		else if (val instanceof Boolean) return new Type.Bool();
		else if (val instanceof StringBuffer) return new Type.Strung();
		else if (val instanceof Character) return new Type.Char();
		else if (val == null) return new Type.Null();
		else InternalFailure.internalFailure("Could not infer constant type", file.filename, constant);
		return null; //unreachable
	}
	private static Type getType(Expr.Cast expression,WyscriptFile file) {
		return expression.getType();
	}
	/**
	 * Returns the type of the expression referenced by the given
	 * indexOF expression.
	 * @param expression
	 * @param file
	 * @return
	 */
	private static Type getType(Expr.IndexOf expression,WyscriptFile file) {
		Type srcType = getType(expression.getSource(),file);
		if (srcType instanceof Type.List) {
			return ((Type.List)srcType).getElement();
		}else {
			InternalFailure.internalFailure("Source type of indexOf is not " +
					"list", file.filename, srcType);
		}
		return null; //unreachable
	}
	/**
	 * Returns the return-type of a "invoke" expression
	 * @param expression
	 * @param file
	 * @return
	 */
	private static Type getType(Expr.Invoke expression,WyscriptFile file) {
		String name = expression.getName();
		List<WyscriptFile.FunDecl> decls = file.functions(name);
		if (decls.size() == 0) {
			InternalFailure.internalFailure("Could not find function with name" +
					" "+name, file.filename, expression);
			return null; //should never happen
		}else {
			WyscriptFile.FunDecl function = decls.get(0);
			return function.ret;
		}
	}
	/**
	 *
	 * @param expression
	 * @param file
	 * @return The type of the list constructor, which is invariably a List with a certain element type
	 */
	private static Type getType(Expr.ListConstructor expression,WyscriptFile file) {
		Expr expr1 = null;
		Expr expr2 = null;
		Type greatest = null;
		for (Expr expr : expression.getArguments()) {
			if (expr1 == null) expr1 = expr;
			else if (expr2 == null) {
				expr2 = expr;
			}else {
				//both expr1 and expr2 are non-null
				Type t1 = getType(expr1, file);
				Type t2 = getType(expr2, file);
				greatest = greaterType(t1, t2, file);
				expr2 = expr1;
				expr2 = expr;
			}
		}
		if (expr1==null && expr2==null) {
			return new Type.List(new Type.Void()); //TODO empty list may not have Void type element
		}else if (expr1 != null && expr2 == null) {
			return new Type.List(getType(expr1,file));
		}else {
			return new Type.List(greatest);
		}
	}
	private static Type getType(Expr.RecordAccess expression,WyscriptFile file) {
		//TODO Implement me!
		return null;

	}
	private static Type getType(Expr.Unary expression,WyscriptFile file) {
		return getType(expression.getExpr(),file);
	}
	/**
	 * Returns the greater of the two types.
	 *
	 * If both t1 and t2 are equivalent, then t1 is returned.
	 * @param t1
	 * @param t2
	 * @param file
	 * @return
	 */
	private static Type greaterType(Type t1, Type t2, WyscriptFile file) {
		if (t1 instanceof Type.Bool && t2 instanceof Type.Bool) {
			return t1;
		} else if (t1 instanceof Type.Char && t2 instanceof Type.Char) {
			return t1;
		} else if (t1 instanceof Type.Int && t2 instanceof Type.Int) {
			return t1;
		} else if (t1 instanceof Type.Real && t2 instanceof Type.Real) {
			return t1;
		} else if (t1 instanceof Type.Strung && t2 instanceof Type.Strung) {
			return t1;
		} else if (t1 instanceof Type.Real && t2 instanceof Type.Int) {
			return t1;
		} else if (t1 instanceof Type.Int && t2 instanceof Type.Real) {
			return t2;
		}
		else if (t1 instanceof Type.List && t2 instanceof Type.List) {
			Type.List l1 = (Type.List) t1;
			Type.List l2 = (Type.List) t2;
			// The following is safe because While has value semantics. In a
			// conventional language, like Java, this is not safe because of
			// references.
			Type element = greaterType(l1.getElement(),l2.getElement(),file);
			if (element == l1.getElement()) return t1;
			else return t2;
		} else {
			syntaxError("expected type " + t1
					+ ", found " + t2, file.filename, t2);
		}
		return null; //unreachable
	}
}
