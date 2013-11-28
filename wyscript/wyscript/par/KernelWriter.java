package wyscript.par;


import java.io.BufferedWriter;
import java.io.File;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;

import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.For;
import wyscript.lang.Type;

/**
 * The first instance of the KernelWriter will take an ordinary for-loop and convert it
 * to Cuda code. There are a number of limitations on what loops can be written. THe first goal
 * is simple loops with <i>int</i> and <i>[int]</i> types, with no nested loops and simple
 * conditionals.
 * @author antunomate
 *
 */

//some challenges
//* determining how to accurately map wyscript onto cuda code
//* facilitating data into the kernel once the analysis is complete
public class KernelWriter {
	private ArrayList<Stmt> body;
	private Stmt.For loop;


	int begin;
	int end;
	int increment;

	List<String> tokens = new ArrayList<String>();

	private boolean kernelable = false;

	public KernelWriter(Stmt.For loop) {
		body = loop.getBody();
		this.loop = loop;
		generateFunctionParameters();
		convertBody(body);
	}
	/**
	 * This method generates a string of function parameters and analyses the
	 * loop body for those assignment statements which require parameters to be
	 * written to kernel.
	 */
	private void generateFunctionParameters() {
		//this requires a scan of the loop body. any external accesses must be
		//parameterised
		for (Stmt statement : loop.getBody()) {
			//what to do here??
			//can analyse variable access
			if (statement instanceof Stmt.Assign) {
				//this is an assignment, must check the LHS and see where if comes from
				Stmt.Assign assign = (Stmt.Assign) statement;
				//how to differentiate between a newly declared variable and an existing one??
				//this information is kept on the frame...
				Expr.LVal left = assign.getLhs();
				Expr right = assign.getRhs();
				//left.
			}
			else if (statement instanceof Stmt.For) {
				//may not want to handle nested loops
			}
			else if (statement instanceof Stmt.IfElse) {

			}else if (statement instanceof Stmt.VariableDeclaration) {
				//this new variable must be added to a collection of internall declared variables
			}else if (statement instanceof Stmt.Print) {
				//could queue up print statements
			}
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

	private void write(Stmt statement) {
		// what happens here?
		if (statement instanceof Stmt.IfElse) {
			write((Stmt.IfElse)statement);
		}
		if (statement instanceof Stmt.VariableDeclaration) {
			write((Stmt.VariableDeclaration) statement);
		}
	}
	private void write(Expr expression) {

	}
	private void write(Stmt.IfElse statement) {
		tokens.add("if");
		tokens.add("(");
		//the condition can only be simple equality, or a statement in boolean
		//logic
		Expr expression = statement.getCondition();
		writeConditional(expression);
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
	private void writeConditional(Expr expression) {
		// TODO Auto-generated method stub

	}
	private void write(Stmt.VariableDeclaration decl) {
		Type type = decl.getType();
		if (type instanceof Type.Int) {
			tokens.add("int");
			tokens.add(decl.getName());
			tokens.add("=");
			//now write the expression
			write(decl.getExpr());
		}
	}
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
}