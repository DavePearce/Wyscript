package wyscript.par.util;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.lang.Expr.IndexOf;
import wyscript.lang.Expr.Variable;
import wyscript.par.KernelRunner;
import wyscript.par.KernelWriter;
import wyscript.util.SyntacticElement;
import wyscript.util.SyntaxError.InternalFailure;

public class LoopModule {
	private Stmt.ParFor loop;

	private List<String> parameters = new ArrayList<String>();
	private Map<String , Type> environment; //passed to kernel writer at runtime
	private Set<String> nonParameterVars = new HashSet<String>();
	private List<Argument> arguments = new ArrayList<Argument>();

	private String fileName;
	private KernelWriter writer;

	private boolean is2D;
	/**
	 * Initialise a KernelWriter which takes <i>name<i/> as its file name and uses
	 * the type mapping given in <i>environment</i> to generate the appropriate kernel
	 * for <i>loop</i>.
	 * @param filename
	 * @param environment
	 * @param loop
	 * @throws IOException
	 * @requires A correct mapping of the symbols used (when the parFor is executed) to their types
	 * @ensures All necessary parameters extracted and converted into a Cuda kernel, as well as stored within KernelWriter
	 */
	public LoopModule(String filename , Map<String , Type> environment , Stmt.ParFor loop){
		this.environment = environment;
		this.fileName = filename;
		loop.getBody();
		this.loop = loop;
		activate();
	}
	private void activate() {
		scanForFunctionParameters(loop.getBody());
		generateArguments();
	}
    /**
	 * This method generates a string of function parameters and analyses the
	 * loop body for those assignment statements which require parameters to be
	 * written to kernel.
	 *
	 * @requires loop != null and loop contains no illegal statements
	 * @ensures that all parameters necessary for a Cuda kernel are stored.
	 */
	private void scanForFunctionParameters(Collection<Stmt> body) {
		//exclude loop index from parameter variables
		nonParameterVars.add(loop.getIndex().getName());
		scanExpr(loop.getSource());
		for (Stmt statement : body) {
		//check for mutabilities in assignment
			if (statement instanceof Stmt.Assign) {
				Stmt.Assign assign = (Stmt.Assign) statement;
				scanAssign(assign);
			}
		//ensure index variable not shadowed
			else if (statement instanceof Stmt.VariableDeclaration) {
				Stmt.VariableDeclaration vardec = (Stmt.VariableDeclaration)statement;
				String name = ((Stmt.VariableDeclaration)statement).getName();
				nonParameterVars.add(name);
				scanExpr(vardec.getExpr());
			}else if (statement instanceof Stmt.IfElse) {
				Stmt.IfElse ifelse = (Stmt.IfElse) statement;
				scanExpr(ifelse.getCondition());
				scanForFunctionParameters(ifelse.getTrueBranch());
				scanForFunctionParameters(ifelse.getFalseBranch());
			}else {
				InternalFailure.internalFailure("Encountered unexpected statement type "
			+statement.getClass(), fileName, statement);
			}
		}
	}
	/**
	 * Scans the lhs and rhs of the assign statement.
	 * @param assign
	 */
	private void scanAssign(Stmt.Assign assign) {
		scanExpr(assign.getLhs());
		scanExpr(assign.getRhs());
	}
	/**
	 * If this expr represents an access to a variable or index, then it is
	 * added to the parameter list
	 * @param expr
	 */
	private void scanExpr(Expr expr) {
		if (expr instanceof Expr.Variable) {
			scanVariableParam((Variable) expr);
		}else if (expr instanceof Expr.Binary) {
			Expr.Binary binary = (Expr.Binary) expr;
			scanExpr(binary.getLhs());
			scanExpr(binary.getRhs());
		}else if (expr instanceof Expr.IndexOf) {
			scanIndexOf((Expr.IndexOf)expr);
		}else if (expr instanceof Expr.Unary) {
			scanExpr(((Expr.Unary) expr).getExpr());
		}
		else {
			//should not have to worry, this expr won't need params
		}
	}
	/**
	 * Writes the actual kernel's function declaration including name and arguments
	 *
	 * @requires The list of parameters to be written is initialised
	 * @ensures The function declaration is written with the required parameters
	 */
	private void generateArguments() {
		for (int i = 0; i < parameters.size() ; i++) {
			String name = parameters.get(i);
			Type type = environment.get(name);
			Argument arg;
			//Argument arg = Argument.convertToArg(name,type);
			if (type instanceof Type.Int) {
				//simply return a single-int argument
				arg = new Argument.SingleInt(name);
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
				}else {
					throw new IllegalArgumentException("Unknown type cannot be converted to kernel argument");
				}
			}
		}

	}
	public String getFuncName() {
		return fileName;
	}
	/**
	 * Add a single variable parameter to parameter list
	 * @param lhs
	 */
	private void scanVariableParam(Variable lhs) {
		if (!parameters.contains(lhs.getName()) &&
				!nonParameterVars.contains(lhs.getName())) parameters.add
				(lhs.getName());
	}
	/**
	 * Add an indexOf operation as parameter. indexOf should be a flat access
	 * to an int value, and this will be checked.
	 * @param indexOf
	 *
	 * @requires indexOf != null and its source to be of Wyscript [int] type
	 * @ensures This parameter added to kernel parameter list
	 */
	private void scanIndexOf(IndexOf indexOf) {
		Expr expression = indexOf.getSource();
		if (expression instanceof Expr.Variable) {
			Expr.Variable srcVar = (Expr.Variable)expression;
			if (!srcVar.getName().equals(loop.getIndex().getName())) {
				//parameters.add(srcVar.getName());
				scanVariableParam(srcVar);
			}
		}else {
			InternalFailure.internalFailure("Expression in index was not " +
					"variable which cannot match loop index", fileName, indexOf);
		}
	}

	public KernelRunner getRunner() {
		return new KernelRunner(this);
	}
	public List<String> getParameters() {
		return new ArrayList<String>(parameters);
	}
	public File getPtxFile() {
		return writer.getPtxFile();
	}
	public SyntacticElement getLoop() {
		return loop;
	}
	public Map<String, Type> getEnvironment() {
		return environment;
	}
	public List<Argument> getArguments() {
		// TODO implement me (when finished configuring arguments)
		return null;
	}
	public boolean is2D() {
		return is2D;
	}
	public boolean isParameter(String name) {
		return parameters.contains(name);
	}
	public Expr.Variable index1() {
		return null;
	}
	public Expr.Variable index2() {
		return null;
	}
}
