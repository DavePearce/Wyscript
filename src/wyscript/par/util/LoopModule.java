package wyscript.par.util;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.ParFor;
import wyscript.lang.Type;
import wyscript.lang.Expr.IndexOf;
import wyscript.lang.Expr.Variable;
import wyscript.par.KernelRunner;
import wyscript.par.KernelWriter;
import wyscript.util.SyntacticElement;
import wyscript.util.SyntaxError.InternalFailure;

public class LoopModule {
	private Stmt.ParFor outerLoop;

	private List<String> parameters = new ArrayList<String>();
	private Map<String , Type> environment; //passed to kernel writer at runtime
	private Set<String> nonParameterVars = new HashSet<String>();
	private List<Argument> arguments = new ArrayList<Argument>();

	private String fileName;
	private KernelWriter writer;

	private boolean is2D;
	private Expr.Variable innerIndex;
	private Expr.Variable outerIndex;

	public final Category category;

	private ParFor innerLoop;

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
		this.outerLoop = loop;
		this.outerIndex = loop.getIndex();
		category = LoopFilter.classify(loop);
		outerIndex = loop.getIndex();
		if (category == Category.IMPINNER) {
			try {
				innerLoop = ((Stmt.ParFor) loop.getBody().get(0));
				innerIndex = ((Stmt.ParFor) loop.getBody().get(0)).getIndex();
			}catch (ClassCastException e) {
				InternalFailure.internalFailure("Expected first statement of loop to be parfor", filename, loop);
			}catch (IndexOutOfBoundsException e) {
				InternalFailure.internalFailure("Explicit-inner loop cannot have empty body", filename, loop);
			}
		}
		activate();
	}
	private void activate() {
		scanForFunctionParameters(outerLoop.getBody());
		generateArguments();
		this.writer = new KernelWriter(this);
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
		nonParameterVars.add(outerLoop.getIndex().getName());
		scanExpr(outerLoop.getSource());
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
			}else if (statement instanceof Stmt.ParFor) {
				Stmt.ParFor loop = (Stmt.ParFor) statement;
				this.innerIndex = loop.getIndex();
				scanExpr(loop.getSource());
				scanForFunctionParameters(loop.getBody());
			}
			else {
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
			scanExpr(binary.getLhs());
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
	 * @param var
	 */
	private void scanVariableParam(Variable var) {
		if (!parameters.contains(var.getName()) &&
				!nonParameterVars.contains(var.getName())
				&& !var.getName().equals(outerIndex) && !var.getName().equals(innerIndex)
				) parameters.add
				(var.getName());
//		Argument arg = new Argument.SingleInt(var.getName());
//		if (!arguments.contains(arg)) arguments.add(arg);

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
		scanExpr(indexOf.getIndex());
		if (expression instanceof Expr.Variable) {
			Expr.Variable srcVar = (Expr.Variable)expression;
			scanVariableParam(srcVar);
		}else if (expression instanceof Expr.IndexOf){
			//scan the next index
			Expr.IndexOf inner = (Expr.IndexOf) expression;
			scanExpr(inner.getIndex());
			if (inner.getSource() instanceof Expr.Variable) {
				scanVariableParam((Variable) inner.getSource());
			}else {
				InternalFailure.internalFailure("Expected variable expression in nested index-of " +
						"variable which cannot match loop index", fileName, indexOf);
			}
		}
		 else {
			InternalFailure.internalFailure("Expression in index was not " +
					"variable which cannot match loop index", fileName, indexOf);
		}
	}

	public KernelRunner getRunner() {
		return new KernelRunner(this);
	}
	public File getPtxFile() {
		return writer.getPtxFile();
	}
	public String getName() {
		return this.fileName;
	}
	public Stmt.ParFor getOuterLoop() {
		return outerLoop;
	}
	public ParFor getInnerLoop() {
		return innerLoop;
	}
	public List<Argument> getArguments() {
		return new ArrayList<Argument>(arguments);
	}
	public Map<String,Type> getEnvironment() {
		return new HashMap<String,Type>(environment);
	}
	public boolean isArgument(String name) {
		for (Argument arg : arguments) {
			if (arg.name.equals(name)) return true;
		}
		return false;
	}
	public Expr.Variable getInnerIndex() {
		return innerIndex;
	}
	public Expr.Variable getOuterIndex() {
		return outerIndex;
	}
}
