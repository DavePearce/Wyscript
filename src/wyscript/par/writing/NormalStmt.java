package wyscript.par.writing;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.lang.Expr.BOp;
import wyscript.lang.Expr.Binary;
import wyscript.lang.Expr.IndexOf;
import wyscript.lang.Expr.Variable;
import wyscript.par.loop.GPULoop;
import wyscript.par.loop.GPUNestedLoop;
import wyscript.util.SyntaxError.InternalFailure;

public class NormalStmt implements KernelStmt{
	private Stmt statement;
	private Map<String,Type> environment;
	private GPULoop gpuLoop;

	public NormalStmt(Map<String,Type> env ,GPULoop loop, Stmt statement) {
		this.statement = statement;
		this.gpuLoop = loop;
		this.environment = env;
	}
	@Override
	public void write(List<String> tokens) {
		write(statement,tokens);
	}

	/**
	 * Convert a single statement to its appropriate kernel form. The statement must
	 * meet certain requirements of for conversion to Cuda code.
	 * @param statement
	 *
	 * @requires statement != null, statement is a legal and the parameters have been initialised
	 * @ensures A single statement of Cuda is written that correctly maps the Wyscript functionality
	 */
	private List<String> write(Stmt statement , List<String> tokens) {
		// what happens here?
		if (statement instanceof Stmt.IfElse) {
			write((Stmt.IfElse)statement,tokens);
		}else if (statement instanceof Stmt.VariableDeclaration) {
			write((Stmt.VariableDeclaration) statement,tokens);
		}else if (statement instanceof Stmt.Assign) {
			write((Stmt.Assign)statement,tokens);
		}else if (statement instanceof Stmt.ParFor) {
			write((Stmt.ParFor)statement,tokens);
		}else if (statement instanceof Stmt.While){
			write((Stmt.While)statement,tokens);
		}else if (statement instanceof Stmt.For) {
//			write((Stmt.For)statement,tokens);
			InternalFailure.internalFailure("Encountered syntactic element not " +
					"supported in parFor loop", "", statement);
		}
		else {
			InternalFailure.internalFailure("Encountered syntactic element not " +
					"supported in parFor loop", "", statement);
		}
		return tokens;
	}
	/**
	 * Writes an assignment statement to the kernel
	 * @param assign
	 *
	 * @requires
	 * @ensures
	 */
	private void write(Stmt.Assign assign, List<String> tokens) {
		Expr.LVal lhs = assign.getLhs();
		write(lhs,tokens);
		tokens.add("=");
		Expr rhs = assign.getRhs();
		write(rhs,tokens);
		tokens.add(";");
	}
	/**
	 * Writes a single expression to the kernel
	 * @param expression
	 *
	 * @requires expression is of an acceptable type and has appropriate parameters
	 * @ensures the Cuda form of the expression is written to the token list
	 */
	private void write(Expr expression,List<String> tokens) {
		if (expression instanceof Expr.ListConstructor) {
			write((Expr.ListConstructor) expression,tokens);
		}
		else if (expression instanceof Expr.LVal) {
			write((Expr.LVal)expression,tokens);
		}else if (expression instanceof Expr.Variable) {
			write((Expr.Variable) expression,tokens);
		}else if (expression instanceof Expr.Constant) {
			write((Expr.Constant) expression,tokens);
		}else if (expression instanceof Expr.IndexOf) {
			write((Expr.IndexOf)expression,tokens);
		}else if (expression instanceof Expr.Binary) {
			write((Expr.Binary) expression,tokens);
		}else if (expression instanceof Expr.Unary) {
			write((Expr.Unary)expression,tokens);
		}else if ((expression instanceof Expr.ListConstructor)) {
			write((Expr.ListConstructor)expression,tokens);
		}
		else{
			InternalFailure.internalFailure("Could not write expression to kernel. Unknown expresion type", "", expression);
		}
	}
	/**
	 * Writes a single Expr.LVal to the kernel.
	 * @param val
	 *
	 * @ensures An expression is written with the correct referencing
	 * and referencing of pointers.
	 */
	private void write(Expr.LVal val,List<String> tokens) {
		if (val instanceof Expr.Variable) {
			//if this is a parameter, have to dereference the pointer
			if (gpuLoop.isArgument(((Expr.Variable)val).getName())) {
					tokens.add("*");
			}
			//write a
			Expr.Variable variable = (Expr.Variable) val;
			//simply add the variable name
			String name = variable.getName();
			Variable loopVar = gpuLoop.getIndexVar();
			if (name.equals(loopVar.getName())) {
				tokens.add(index1D());
			}else {
				tokens.add(name);
			}
		}else if (val instanceof Expr.IndexOf) {
			write((Expr.IndexOf)val,tokens);
		}

	}
	/**
	 * Here a nested loop is being written
	 * @param statement
	 * @param tokens
	 */
	private void write(Stmt.ParFor statement, List<String> tokens) {
		//consider the expressions of the loop...
		//the kernel runner should take care of the thread running...
		for (Stmt stmt : statement.getBody()) {
			write(stmt,tokens);
		}

	}
	private String index1D() {
		return indexName1D;
	}
	private void write(Expr.Binary binary,List<String> tokens) {
		tokens.add("(");
		write(binary.getLhs(),tokens);
		tokens.add(")");
		writeOp(binary.getOp(),tokens);
		tokens.add("(");
		write(binary.getRhs(),tokens);
		tokens.add(")");
	}
	private void write(Expr.Unary unary,List<String> tokens) {
		switch (unary.getOp()) {
		case LENGTHOF:
			writeLengthOf(unary.getExpr(),tokens);
			break;
		case NEG:
			tokens.add("-");
			tokens.add("(");
			write(unary.getExpr(),tokens);
			tokens.add(")");
			break;
		case NOT:
			tokens.add("!");
			tokens.add("(");
			write(unary.getExpr(),tokens);
			tokens.add(")");
			break;
		default:
			InternalFailure.internalFailure("Unknown unary expression encountered"
					, "", unary);

		}
	}
	private void writeLengthOf(Expr expr,List<String> tokens) {
		if (expr instanceof Expr.Variable) {
			String name = ((Expr.Variable)expr).getName();
			tokens.add("(");
			tokens.add("*");
			String lengthName = gpuLoop.lengthName(name);
			tokens.add(lengthName);
			tokens.add(")");
		}
		else if (expr instanceof Expr.ListConstructor) {
			int size = ((Expr.ListConstructor) expr).getArguments().size();
			tokens.add(Integer.toString(size));
		}else if (expr instanceof Expr.Binary) {
			Expr.Binary binary = (Binary) expr;
			if (binary.getOp().equals(BOp.RANGE)) {
				tokens.add("(");
				write(binary.getRhs(),tokens);
				tokens.add("-");
				write(binary.getLhs(),tokens);
				tokens.add(")");
			}
		}
		else {
			//TODO Implement me
			InternalFailure.internalFailure("Writing length of this expression not implemented", "", expr);
		}
	}
	private void writeOp(BOp op,List<String> tokens) {
		tokens.add(op.toString());
	}
	/**
	 * Writes a single constant to token list. This constant may be an int only
	 * @param constant
	 *
	 * @requires the constant is an int
	 */
	private void write(Expr.Constant constant,List<String> tokens) {
		Object val = constant.getValue();
		if (val instanceof Integer) {
			tokens.add(Integer.toString((Integer)val));
		}else {
			InternalFailure.internalFailure("Cannot write this constant: "+val,
					"", constant);
		}
	}
	/**
	 * This method is not implemented yet.
	 *
	 * Writes a list constructor to kernel code.
	 * @param list
	 */
	private void write(Expr.ListConstructor list,List<String> tokens) {
		tokens.add("{");
		boolean needComma = false;
		for (Expr expr : list.getArguments()) {
			if (needComma) tokens.add(",");
			write(expr,tokens);
			needComma = true;
		}
		tokens.add("}");
/*		InternalFailure.internalFailure("Writing list constructors not implemented"
				, fileName, list);*/
	}
	/**
	 * Write a single indexOf operation to token list
	 * @param expr
	 */
	private void write(Expr.IndexOf indexOf, List<String> tokens) {
		Expr src = indexOf.getSource();
		if (indexOf.getSource() instanceof Expr.Variable) {
			Expr indexVar = indexOf.getIndex();
			//indexVar is an instance of [int]
			//source expression must be of type...
			if (src instanceof Expr.Variable) {
				write1DIndexOf(tokens, (Expr.Variable) src, indexVar);
			}
			else {
				InternalFailure.internalFailure("Can only perform indexof on list", "", indexOf);
			}
		}else if (src instanceof Expr.IndexOf) {
			write2DIndexOf(tokens, indexOf);
		}else {
			InternalFailure.internalFailure("Expected source type to be of type list", "", indexOf.getSource());
		}
	}
	/**
	 *
	 * @param tokens
	 * @param indexOf
	 * @param outer
	 */
	private void write2DIndexOf(List<String> tokens, Expr.IndexOf indexOf) {
		Expr indexSrc = indexOf.getSource();
		Expr outerIndex = indexOf.getIndex();
		if (indexSrc instanceof Expr.IndexOf) {
			//the source is an index!
			Expr innerSrc = ((Expr.IndexOf) indexOf).getSource();
			//check if this indeed a nested indexof operation
			if (innerSrc instanceof Expr.IndexOf) {
				Expr innerInnerSrc = ((IndexOf) innerSrc).getSource();
				Expr innerIndex = ((Expr.IndexOf) innerSrc).getIndex();
				//the finally-indexed value must be a variable
				if (innerInnerSrc instanceof Expr.Variable) {
					Expr.Variable variable = (Variable) innerInnerSrc;
					//now have the inner src which is a variable
					//time to check out the indices
					if (outerIndex instanceof Expr.Variable && innerIndex
							instanceof Expr.Variable && gpuLoop instanceof GPUNestedLoop) {
						String innerName = ((Expr.Variable) outerIndex).getName();
						String outerName = ((Expr.Variable) innerIndex).getName();
//						if (gpuLoop instanceof GPUNestedLoop) {
						GPUNestedLoop nestedGPULoop = (GPUNestedLoop)gpuLoop;
						if (outerName.equals(nestedGPULoop.getIndexVar().getName())&&
								innerName.equals(nestedGPULoop.getInnerIndexVar().getName())){
							//cleared for writing a 2d array access
							//first write the variable
							tokens.add(variable.getName());
							tokens.add("[");
							tokens.add(index2D());
							tokens.add("]");
						}else {
							InternalFailure.internalFailure("2D indices must match both loop indices", "", indexOf);
						}
//						}
//						else {
//							//#omg #error
//							InternalFailure.internalFailure("Non-nested loop detected while writing 2D indexof", name, indexOf);
//						}
					}
					else {
						//InternalFailure.internalFailure("Writing non-loop index not implemented", name, indexOf);
						//TODO write complex indexof formula here
						tokens.add(variable.getName());
						tokens.add("[");
						write(outerIndex, tokens);
						tokens.add("*");
						tokens.add(
							gpuLoop.widthName(((Expr.Variable) innerInnerSrc).getName()));
						tokens.add("+");
						write(innerIndex,tokens);
						tokens.add("]");
					}
				}else {
					InternalFailure.internalFailure("IndexOf indices must both be nested loop indices", "", indexOf);
				}
			}else {
				InternalFailure.internalFailure("For 2D IndexOf, inner src must be indexof", "", indexOf);
			}
		}else {
			InternalFailure.internalFailure("For 2D IndexOf, outer src must be indexof", "", indexOf);
		}
	}

	private String index2D() {
		return indexName2D;
	}
	/**
	 *
	 * @param tokens
	 * @param src
	 * @param indexVar
	 */
	private void write1DIndexOf(List<String> tokens, Expr.Variable src, Expr indexVar) {
		Type typeOfVar = environment.get(src.getName());
		Type listType = ((Type.List)typeOfVar).getElement();
		if (listType instanceof Type.Int) {
			//the type is correct for a kernel, write it here
			tokens.add(src.getName());
			if (indexVar instanceof Expr.Variable) {
				if (((Expr.Variable) indexVar).getName().equals(gpuLoop.getIndexVar().getName())) {
					tokens.add("["+index1D()+"]");
				}
			}else if (indexVar instanceof Expr.Constant) {
				tokens.add("[");
				write((Expr.Constant)indexVar,tokens);
				tokens.add("]");
			}else {
				InternalFailure.internalFailure("Index should be parFor loop index or constant", "", src);
			}
		}
		else{
			InternalFailure.internalFailure("List type should be int for kernel conversion", "", src);
		}
	}
	/**
	 * Writes a classical conditional statement to the kernel
	 * @param statement
	 */
	private void write(Stmt.IfElse statement,List<String> tokens) {
		tokens.add("if");
		tokens.add("(");
		//the condition can only be simple equality, or a statement in boolean
		//logic
		Expr expression = statement.getCondition();
		write(expression,tokens);
		tokens.add(")");
		tokens.add("{");
		//branches may be empty
		for (Stmt s : statement.getTrueBranch()) {
			write(s,tokens); //write the single statement
		}
		tokens.add("}");
		tokens.add("else");
		tokens.add("{");
		for (Stmt s : statement.getFalseBranch()) {
			write(s,tokens); //write the single statement
		}
		tokens.add("}");
	}
	/**
	 * Writes a single variable declaration to the kernel.
	 * @param decl
	 */
	private void write(Stmt.VariableDeclaration decl,List<String> tokens) {
		Type type = decl.getType();
		if (type instanceof Type.Int) {
			tokens.add("int");
			tokens.add(decl.getName());
			tokens.add("=");
			//now write the expression
			write(decl.getExpr(),tokens);
			tokens.add(";");
		}else if (type instanceof Type.List) {
			Type element = ((Type.List) type).getElement();
			if (element instanceof Type.Int && decl.getExpr() instanceof Expr.ListConstructor) {
				List<Expr> elements = ((Expr.ListConstructor)decl.getExpr()).getArguments();
				tokens.add("int");
				tokens.add(decl.getName());
				tokens.add("["+elements.size()+"]");
				tokens.add("=");
				write((Expr.ListConstructor)decl.getExpr(),tokens);
				tokens.add(";");
			}else {
				InternalFailure.internalFailure("Can only write explicit list of integers","",decl);
			}
		}
		else {
			InternalFailure.internalFailure("Cannot write variable declaration for the given type","",decl);

		}
		environment.put(decl.getName(), type);
	}
	private void write(Stmt.While whileloop , List<String> tokens) {
		tokens.add("while");
		tokens.add("(");
		write(whileloop.getCondition(),tokens);
		tokens.add(")");
		tokens.add("{");
		for (Stmt statement : whileloop.getBody()) {
			write(statement, tokens);
		}
		tokens.add("}");
	}
	private void write(Stmt.For forloop , List<String> tokens) {
		//for
		Expr src = forloop.getSource();
		tokens.add("for");
		tokens.add("(");
		tokens.add("int");
		tokens.add("INDEX");
		tokens.add("=");
		tokens.add("0");
		tokens.add(";");
		tokens.add("INDEX");
		tokens.add("<");
		writeLengthOf(src, tokens);
		tokens.add(";");
		tokens.add("INDEX++");
		tokens.add(")");
		tokens.add("{");
		for (Stmt statement : forloop.getBody()) {
			write(statement, tokens);
		}
		tokens.add("}");
	}

}
