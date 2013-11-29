package wyscript.par;

import java.io.BufferedWriter;
import java.io.File;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import wyscript.lang.Expr;
import wyscript.lang.Expr.*;
import wyscript.lang.Expr.IndexOf;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.For;
import wyscript.lang.Type;
import wyscript.lang.WyscriptFile;
import wyscript.util.SyntaxError.InternalFailure;

/**
 * The first instance of the KernelWriter will take an ordinary for-loop and convert it
 * to Cuda code. There are a number of limitations on what loops can be written. THe first goal
 * is simple loops with <i>int</i> and <i>[int]</i> types, with no nested loops and simple
 * conditionals. The kernel writer will compile the cuda code and return an error if this operation
 * is not successful.
 * @author antunomate
 *
 */

//some challenges
//* determining how to accurately map wyscript onto cuda code
//* facilitating data into the kernel once the analysis is complete
//*converting between cuda and wyscript types
public class KernelWriter {
	private ArrayList<Stmt> body;
	private Stmt.For loop;

	int begin;
	int end;
	int increment;

	List<String> tokens = new ArrayList<String>();
	List<String> parameters;
	private boolean kernelable = true;

	public KernelWriter(WyscriptFile file , Stmt.For loop) {
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
				}

				write(left);
			}
		}
	}
	/**
	 * Add an indexOf operation as parameter. indexOf should be a flat access
	 * to an int value
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
				//TODO add an internal failure here
			}
		}else {
			//TODO A graceful way for this (user?) error to be dealt with
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
		else if (statement instanceof Stmt.VariableDeclaration) {
			write((Stmt.VariableDeclaration) statement);
		}else if (statement instanceof Stmt.Assign) {
			write((Stmt.Assign)statement);
		}
	}
	/**
	 *
	 * @param assign
	 */
	private void write(Stmt.Assign assign) {
		Expr lhs = assign.getLhs();
		write(lhs);
		tokens.add("=");
		Expr rhs = assign.getLhs();
		write(rhs);
		tokens.add(";");
	}

	private void write(Expr expression) {
		if (expression instanceof Expr.IndexOf) {

		}else if (expression instanceof Expr.IndexOf) {

		}
	}
	private void write(Expr.LVal val) {
		//val.g
	}
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
	private void writeCondition(Expr expression) {
		// TODO Auto-generated method stub
		//if (expression instanceof Expr.)
	}
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