package wyscript.par;

import java.io.BufferedWriter;
import java.io.File;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import wyscript.lang.Expr;
import wyscript.lang.Expr.*;
import wyscript.lang.Expr.IndexOf;
import wyscript.lang.Expr.Variable;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.For;
import wyscript.lang.Type;
import wyscript.lang.WyscriptFile;
import wyscript.util.SyntaxError.InternalFailure;

/**
 * The first instance of the KernelWriter will take an ordinary for-loop and convert it
 * to Cuda code. There are a number of limitations on what loops can be written. The first goal
 * is simple loops with <i>int</i> and <i>[int]</i> types, with no nested loops and simple
 * conditionals. The kernel writer will compile the cuda code and return an error if this operation
 * is not successful.
 * @author antunomate
 *
 */
public class KernelWriter {
	private ArrayList<Stmt> body;
	private Stmt.ParFor loop;

	private int begin;
	private int end;
	private int increment;

	private List<String> tokens = new ArrayList<String>();
	private List<String> parameters = new ArrayList<String>();

	private Map<String , Type> environment; //passed to kernel writer at runtime
	//this is the mapping from array to array length
	private Map<String,String> lengthFunction = new HashMap<String,String>();
	public Map<String, Type> getEnvironment() {
		return environment;
	}
	private String indexName = "i";
	/**
	 * Initialise a KernelWriter which takes <i>name<i/> as its file name and uses
	 * the type mapping given in <i>environment</i> to generate the appropriate kernel
	 * for <i>loop</i>.
	 * @param name
	 * @param environment
	 * @param loop
	 */
	public KernelWriter(String name , Map<String , Type> environment , Stmt.ParFor loop) {
		this.environment = environment;
		this.body = loop.getBody();
		this.loop = loop;
		generateFunctionParameters();
		writeFunctionDeclaration();
		tokens.add("{");
		convertBody(body);
		tokens.add("}");
	}
	/**
	 * This method generates a string of function parameters and analyses the
	 * loop body for those assignment statements which require parameters to be
	 * written to kernel.
	 *
	 * This function @ensures that all parameters necessary for a Cuda kernel are
	 * stored.
	 */
	private void generateFunctionParameters() {
		//scan the loop body, determine what must be added as parameter
		for (Stmt statement : loop.getBody()) {
			//an assign statement implies a mutable change
			if (statement instanceof Stmt.Assign) {
				//this is an assignment, must check the LHS and see where if comes from
				Stmt.Assign assign = (Stmt.Assign) statement;
				//assign.g
				Expr.LVal left = assign.getLhs();
				if (assign.getLhs() instanceof Expr.IndexOf) {
					addIndexOfParam((Expr.IndexOf)assign.getLhs());
				}else if (assign.getLhs() instanceof Expr.Variable) {
					addVariableParam((Expr.Variable)assign.getLhs());
				}//TODO add other case of assignment here

				//write(left);
			}
		}
	}
	/**
	 * Writes the actual kernel's function declaration including name and arguments
	 */
	private void writeFunctionDeclaration() {
		tokens.add("__global__");
		tokens.add("void");
		tokens.add("(");
		for (int i = 0; i < parameters.size() ; i++) {
			String name = parameters.get(i);
			//now work out the type of each parameters
			Type type = environment.get(name);
			//write an array pointer, and also give a parameter for length
			if (type instanceof Type.List) {
				Type.List list = (Type.List) type;
				if (list.getElement() instanceof Type.Int) {
					tokens.add("int*");
					tokens.add(name);
					//note that the length is added to the list parameter
					tokens.add(",");
					tokens.add("int");
					//qualify length of array with '_length'
					tokens.add(name + "_length");
				}else {
					InternalFailure.internalFailure("List type should be int for kernel conversion", null, list);
				}
			}
			//TODO WARNING potential off-by-one error
			if (i>=1 && i < parameters.size()-2) {
				tokens.add(",");
			}
		}
		tokens.add(")");
	}
	/**
	 *
	 * @param lhs
	 */
	private void addVariableParam(Variable lhs) {
		parameters.add(lhs.getName());
	}
	/**
	 * Add an indexOf operation as parameter. indexOf should be a flat access
	 * to an int value, and this will be checked.
	 * @param indexOf
	 */
	private void addIndexOfParam(IndexOf indexOf) {
		Expr expression = indexOf.getSource();
		if (indexOf.getIndex().equals(loop.getIndex())) { //TODO verify whether it is correct to compare these expressions
			//now need to get name of source expression
			if (expression instanceof Expr.Variable) {
				Expr.Variable srcVar = (Expr.Variable)expression;
				String name = srcVar.getName();
				//add name to parameter list only
				parameters.add(name);
			}else {
				InternalFailure.internalFailure("Source expression in index of was not variable", null, expression);
			}
		}else {
			InternalFailure.internalFailure("Expression in index did not match loop index", null, indexOf);
		}
	}
	/**
	 * Maps the body of the loop onto Cuda code
	 * @param body2
	 */
	private void convertBody(ArrayList<Stmt> body2) {
		for (Stmt statement : body) {
			write(statement);
		}
	}
	/**
	 * Convert a single statement to its appropriate kernel form. The statement must
	 * meet certain requirements of for conversion to Cuda code.
	 * @param statement
	 */
	private void write(Stmt statement) {
		// what happens here?
		if (statement instanceof Stmt.IfElse) {
			write((Stmt.IfElse)statement);
		}
		else if (statement instanceof Stmt.VariableDeclaration) {
			write((Stmt.VariableDeclaration) statement);
		}else if (statement instanceof Stmt.Assign) {
			write((Stmt.Assign)statement);
		}
	}
	/**
	 * Writes an assignment statement to the kernel
	 * @param assign
	 */
	private void write(Stmt.Assign assign) {
		Expr.LVal lhs = assign.getLhs();
		write(lhs);
		tokens.add("=");
		Expr rhs = assign.getLhs();
		write(rhs);
		tokens.add(";");
	}
	/**
	 * Writes a single expression to the kernel
	 * @param expression
	 */
	private void write(Expr expression) {
		//TODO What is wrong here? Are there subtypes of expression who overload this method and bypass it completely.
	}
	/**
	 * Writes a single Expr.LVal to the kernel.
	 * @param val
	 */
	private void write(Expr.LVal val) {
		//val.g
		if (val instanceof Expr.Variable) {
			//write a
			Expr.Variable variable = (Expr.Variable) val;
			//simply add the variable name
			tokens.add(variable.getName());
		}else if (val instanceof Expr.IndexOf) {
			writeIndexOf(val);
		}

	}
	/**
	 * Checks whether an indexOf Expr.LVal left-hand expression matched the
	 * correct type for kernel conversion. Then writes it to the token list.
	 * @param val
	 */
	private void writeIndexOf(Expr.LVal val) {
		Expr.IndexOf indexOf = (Expr.IndexOf) val;
		if (indexOf.getSource() instanceof Expr.Variable) {
			Expr.Variable indexVar = (Expr.Variable)indexOf.getSource();
			//indexVar is an instance of [int]
			//source expression must be of type...
			Type typeOfVar = environment.get(indexVar.getName());
			if (typeOfVar instanceof Type.List) {
				Type listType = ((Type.List)typeOfVar).getElement();
				if (listType instanceof Type.Int) {
					//the type is correct for a kernel, write it here
					tokens.add(indexVar.getName());
					tokens.add("["+indexName+"]");
				}else{
					InternalFailure.internalFailure("List type should be int for kernel conversion", null, indexVar);
				}
			}

		}else {
			InternalFailure.internalFailure("Expected source type to be of type list", null, indexOf.getSource());
		}
		if (indexOf.getIndex().equals(loop.getIndex())) { //TODO Potential issue with comparing indices

		}
	}
	/**
	 * Writes a classical conditional statement to the kernel
	 * @param statement
	 */
	private void write(Stmt.IfElse statement) {
		tokens.add("if");
		tokens.add("(");
		//the condition can only be simple equality, or a statement in boolean
		//logic
		Expr expression = statement.getCondition();
		writeCondition(expression);
		tokens.add(")");
		//branches may be empty
		for (Stmt s : statement.getTrueBranch()) {
			write(s); //write the single statement
		}
		tokens.add("}");
		tokens.add("else");
		tokens.add("{");
		for (Stmt s : statement.getFalseBranch()) {
			write(s); //write the single statement
		}
		tokens.add("}");
	}
	/**
	 * Writes a single condition expression without brackets
	 * @param expression
	 */
	private void writeCondition(Expr expression) {
		// TODO Implement me
	}
	/**
	 * Writes a single variable declaration to the kernel.
	 * @param decl
	 */
	private void write(Stmt.VariableDeclaration decl) {
		Type type = decl.getType();
		if (type instanceof Type.Int) {
			tokens.add("int");
			tokens.add(decl.getName());
			tokens.add("=");
			//now write the expression
			write(decl.getExpr());
			tokens.add(";");
		}else {
			InternalFailure.internalFailure("Cannot write variable declaration for the given type",null,null);

		}
	}
	/**
	 * Return the File object associated with this kernel
	 * @return
	 */
	public File getPtxFile() {
		return null;
	}
	/**
	 * Returns a List of the string representation of the kernel writer's tokens
	 * @return
	 */
	public List<String> getTokenList() {
		List<String> output = new ArrayList<String>(tokens);
		return output;
	}
	@Override
	public String toString() {
		return tokens.toString();
	}
	public KernelRunner getRunner() {
		return null;
		// TODO Auto-generated method stub

	}
}