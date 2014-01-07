package wyscript.par.loop;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.ParFor;
import wyscript.lang.Type;
import wyscript.lang.Expr.IndexOf;
import wyscript.lang.Expr.Variable;
import wyscript.par.util.Argument;
import wyscript.util.SyntaxError.InternalFailure;

public class GPUUtils {
	private GPUUtils(){}

	public static List<Argument> scanForFunctionParameters(ParFor loop,
			List<String> parameters, Set<String> nonparameters,
			HashMap<String, Type> env) {
		return scanForFunctionParameters(loop.getBody(), parameters, nonparameters, env);
	}

    /**
	 * This method generates a string of function parameters and analyses the
	 * loop body for those assignment statements which require parameters to be
	 * written to kernel.
     * @param nonparameters2
     * @param parameters2
     * @param nonParameterVars
	 *
	 * @requires loop != null and loop contains no illegal statements
	 * @ensures that all parameters necessary for a Cuda kernel are stored.
	 */
	public static List<Argument> scanForFunctionParameters(Collection<Stmt> body,
			List<String> parameters, Set<String> nonparameters, Map<String,Type> env) {
		//exclude loop index from parameter variables
		for (Stmt statement : body) {
		//check for mutabilities in assignment
			if (statement instanceof Stmt.Assign) {
				Stmt.Assign assign = (Stmt.Assign) statement;
				scanAssign(assign,parameters,nonparameters);
			}
		//ensure index variable not shadowed
			else if (statement instanceof Stmt.VariableDeclaration) {
				Stmt.VariableDeclaration vardec = (Stmt.VariableDeclaration)statement;
				String name = ((Stmt.VariableDeclaration)statement).getName();
				nonparameters.add(name);
				scanExpr(vardec.getExpr(),parameters,nonparameters);
			}else if (statement instanceof Stmt.IfElse) {
				Stmt.IfElse ifelse = (Stmt.IfElse) statement;
				scanExpr(ifelse.getCondition(),parameters,nonparameters);
				scanForFunctionParameters(ifelse.getTrueBranch(),parameters,nonparameters,env);
				scanForFunctionParameters(ifelse.getFalseBranch(),parameters,nonparameters,env);
			}else if (statement instanceof Stmt.ParFor) {
				InternalFailure.internalFailure("ParFor statement forbidden in ParFor body", "", statement);
			}else if (statement instanceof Stmt.For) {
				Stmt.For loop = (Stmt.For) statement;
				scanExpr(loop.getSource(),parameters,nonparameters);
				scanForFunctionParameters(loop.getBody(),parameters,nonparameters,env);
			}else if (statement instanceof Stmt.While) {
				Stmt.While loop = (Stmt.While) statement;
				scanExpr(loop.getCondition(),parameters,nonparameters);
				scanForFunctionParameters(loop.getBody(),parameters,nonparameters,env);
			}
			else {
				InternalFailure.internalFailure("Encountered unexpected statement type "
			+statement.getClass(), "UNKNOWN FILE", statement);
			}
		}
		return convertToArgList(parameters,env);
	}
	/**
	 * Takes a list of parameter names and the environment map and
	 * returns a list of corresponding arguments
	 * @param parameters
	 * @param env
	 * @return List of Argument-type
	 */
	private static List<Argument> convertToArgList(List<String> parameters, Map<String,Type> env) {
		List<Argument> arguments = new ArrayList<Argument>();
		for (int i = 0; i < parameters.size() ; i++) {
			String name = parameters.get(i);
			Type type = env.get(name);
			Argument arg;
			//Argument arg = Argument.convertToArg(name,type);
			if (type instanceof Type.Int) {
				//simply return a single-int argument
				arg = new Argument.SingleInt(name);
				arguments.add(arg);
			}else if (type instanceof Type.List) {
				//differentiate between 1D and 2D lists
				Type elementType = (((Type.List) type).getElement());
				if (elementType instanceof Type.Int) {
					arg = new Argument.List1D(name);
					arguments.add(arg);
					arguments.add(new Argument.Length1D(name));
				}else if (elementType instanceof Type.List) {
					if (((Type.List) elementType).getElement() instanceof Type.Int) {
						arg = new Argument.List2D(name);
						arguments.add(arg);
						//add height first
						arguments.add(new Argument.Length2D(name,true));
						arguments.add(new Argument.Length2D(name,false));
					}
				}else if (elementType instanceof Type.Real) {
					arg = new Argument.DoubleList1D(name);
					arguments.add(arg);
					arguments.add(new Argument.Length1D(name));
				}
				else {
					throw new IllegalArgumentException("Unknown type cannot be converted to kernel argument");
				}
			}else if (type instanceof Type.Real) {
				arg = new Argument.SingleDouble(name);
				arguments.add(arg);
			}
		}
		return arguments;
	}

	/**
	 * Scans the lhs and rhs of the assign statement.
	 * @param assign
	 * @param nonparameters
	 * @param parameters
	 */
	private static void scanAssign(Stmt.Assign assign, List<String> parameters, Set<String> nonparameters) {
		scanExpr(assign.getLhs(), parameters, nonparameters);
		scanExpr(assign.getRhs(), parameters, nonparameters);
	}
	/**
	 * If this expr represents an access to a variable or index, then it is
	 * added to the parameter list
	 * @param expr
	 * @param nonparameters
	 * @param parameters
	 */
	private static void scanExpr(Expr expr, List<String> parameters, Set<String> nonparameters) {
		if (expr instanceof Expr.Variable) {
			scanVariableParam((Variable) expr, parameters, nonparameters);
		}else if (expr instanceof Expr.Binary) {
			Expr.Binary binary = (Expr.Binary) expr;
			scanExpr(binary.getLhs(), parameters, nonparameters);
			scanExpr(binary.getRhs(), parameters, nonparameters);
		}else if (expr instanceof Expr.IndexOf) {
			scanIndexOf((Expr.IndexOf)expr,parameters,nonparameters);
		}else if (expr instanceof Expr.Unary) {
			scanExpr(((Expr.Unary) expr).getExpr(), parameters, nonparameters);
		}else if (expr instanceof Expr.Cast) {
			scanExpr(((Expr.Cast) expr).getSource(),parameters,nonparameters);
		}
		else {
			//should not have to worry, this expr won't need params
		}
	}
	/**
	 * Add a single variable parameter to parameter list
	 * @param var
	 * @param parameters
	 * @param nonparameters
	 */
	private static void scanVariableParam(Variable var, List<String> parameters, Set<String> nonparameters) {
		String name = var.getName();
		if (!(parameters.contains(name)||nonparameters.contains(name))) {
			parameters.add(name);
		}
	}
	/**
	 * Add an indexOf operation as parameter. indexOf should be a flat access
	 * to an int value, and this will be checked.
	 * @param indexOf
	 * @param nonparameters
	 * @param parameters
	 *
	 * @requires indexOf != null and its source to be of Wyscript [int] type
	 * @ensures This parameter added to kernel parameter list
	 */
	private static void scanIndexOf(IndexOf indexOf, List<String> parameters, Set<String> nonparameters) {
		Expr expression = indexOf.getSource();
		scanExpr(indexOf.getIndex(), parameters, nonparameters);
		if (expression instanceof Expr.Variable) {
			Expr.Variable srcVar = (Expr.Variable)expression;
			scanVariableParam(srcVar, parameters,nonparameters);
		}else if (expression instanceof Expr.IndexOf){
			//scan the next index
			Expr.IndexOf inner = (Expr.IndexOf) expression;
			scanExpr(inner.getSource(), parameters, nonparameters);
			if (inner.getSource() instanceof Expr.Variable) {
				scanVariableParam((Variable) inner.getSource(), parameters, nonparameters);
			}else {
				InternalFailure.internalFailure("Expected variable expression in nested index-of " +
						"variable which cannot match loop index", "", indexOf);
			}
		}
		 else {
			InternalFailure.internalFailure("Expression in index was not " +
					"variable which cannot match loop index", "", indexOf);
		}
	}

}
