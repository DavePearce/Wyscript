package wyscript.io;

import static wyscript.util.SyntaxError.internalFailure;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import wyscript.Interpreter;
import wyscript.lang.*;
import wyscript.util.*;

/**
 * An extended interpreter - instead of outputting directly to the
 * console, converts a given WyScriptFile into an equivalent JavaScript
 * one - doing so requires partial interpretation of some WyScript code.
 *
 */
public class JavaScriptFileWriter {
	private PrintStream out;
	private WyscriptFile file;
	
	public JavaScriptFileWriter(File file) throws IOException {
		this.out = new PrintStream(new FileOutputStream(file));
	}

	public void close() {
		out.close();
	}

	public void write(WyscriptFile wf) {
		this.file = wf;
		userTypes = new HashMap<String, Type>();

		//Need to create additional helper functions
		setupFunctions();

		//Next, sort out constants and named types
		for (WyscriptFile.Decl declaration : wf.declarations) {
			if (declaration instanceof WyscriptFile.ConstDecl) {
				write((WyscriptFile.ConstDecl)declaration);
			}

			else if (declaration instanceof WyscriptFile.TypeDecl) {
				WyscriptFile.TypeDecl td = (WyscriptFile.TypeDecl)declaration;
				userTypes.put(td.name(), td.type);
			}
		}

		for(WyscriptFile.Decl declaration : wf.declarations) {
			if(declaration instanceof WyscriptFile.FunDecl) {
				write((WyscriptFile.FunDecl) declaration);
			}
		}
	}
	
	public void write(WyscriptFile.FunDecl fd) {
		out.print("function " + fd.name + "(");
		boolean firstTime = true;

		for(WyscriptFile.Parameter p : fd.parameters) {
			if(!firstTime) {
				out.print(", ");
			}
			firstTime=false;
			out.print(p.name);
		}
		out.println(") {");
		write(fd.statements, 1);
		out.println("}");
	}

	public void write(List<Stmt> statements, int indent) {
		for(Stmt s : statements) {
			write(s, indent);
		}
	}

	public void write(Stmt stmt, int indent) {
		if(stmt instanceof Stmt.Atom) {
			indent(indent);
			writeAtom((Stmt.Atom) stmt);
			out.println(";");
		} else if(stmt instanceof Stmt.IfElse) {
			write((Stmt.IfElse) stmt, indent);
		} else if(stmt instanceof Stmt.OldFor) {
			write((Stmt.OldFor) stmt, indent);
		} else if(stmt instanceof Stmt.While) {
			write((Stmt.While) stmt, indent);
		} else if (stmt instanceof Stmt.For) {
			write((Stmt.For) stmt, indent);
		}
		else {
			internalFailure("unknown statement encountered (" + stmt + ")", file.filename,stmt);
		}
	}

	public void write(Stmt.IfElse stmt, int indent) {
		indent(indent);
		out.print("if(");
		write(stmt.getCondition());
		out.println(") {");
		write(stmt.getTrueBranch(),indent+1);
		if(stmt.getFalseBranch().size() > 0) {
			indent(indent);
			out.println("} else {");
			write(stmt.getFalseBranch(),indent+1);
		}
		indent(indent);
		out.println("}");
	}

	public void write(Stmt.OldFor stmt, int indent) {
		indent(indent);
		out.print("for(");
		writeAtom(stmt.getDeclaration());
		out.print(";");
		write(stmt.getCondition());
		out.print(";");
		writeAtom(stmt.getIncrement());
		out.println(") {");
		write(stmt.getBody(),indent+1);
		indent(indent);
		out.println("}");
	}
	
	public void write(Stmt.While stmt, int indent) {
		indent(indent);
		out.print("while(");
		write(stmt.getCondition());
		out.println(") {");
		write(stmt.getBody(),indent+1);
		indent(indent);
		out.println("}");
	}

	public void writeAtom(Stmt stmt) {
		if(stmt instanceof Stmt.Assign) {
			write((Stmt.Assign) stmt);
		} else if(stmt instanceof Stmt.Print) {
			write((Stmt.Print) stmt);
		} else if(stmt instanceof Stmt.Return) {
			write((Stmt.Return) stmt);
		} else if(stmt instanceof Stmt.VariableDeclaration) {
			write((Stmt.VariableDeclaration) stmt);
		} else {
			internalFailure("unknown statement encountered (" + stmt + ")", file.filename,stmt);
		}
	}

	public void write(Stmt.Assign stmt) {
		write(stmt.getLhs());
		out.print(" = ");
		write(stmt.getRhs());
	}


	public void write(Stmt.Print stmt) {
		Type type = stmt.getExpr().attribute(Attribute.Type.class).type;
		out.print("sysout.println(");
		if(type instanceof Type.Int) { 
			out.print("Math.round(");
			write(stmt.getExpr());
			out.print(")");
		} else {
			write(stmt.getExpr());
		}
		out.print(")");
	}

	public void write(Stmt.Return stmt) {
		Expr expr = stmt.getExpr();
		out.print("return");
		if(expr != null) {
			out.print(" ");
			write(expr);
		}
	}

	public void write(Stmt.VariableDeclaration stmt) {
		Expr init = stmt.getExpr();
		out.print("var " + stmt.getName());
		if(init != null) {
			out.print(" = ");
			write(init);
		}
	}

	public void write(Expr expr) {
		if(expr instanceof Expr.Binary) {
			write((Expr.Binary) expr);
		} else if(expr instanceof Expr.Cast) {
			write((Expr.Cast) expr);
		} else if(expr instanceof Expr.Constant) {
			write((Expr.Constant) expr);
		} else if(expr instanceof Expr.IndexOf) {
			write((Expr.IndexOf) expr);
		} else if(expr instanceof Expr.Invoke) {
			write((Expr.Invoke) expr);
		} else if(expr instanceof Expr.ListConstructor) {
			write((Expr.ListConstructor) expr);
		} else if(expr instanceof Expr.RecordAccess) {
			write((Expr.RecordAccess) expr);
		} else if(expr instanceof Expr.RecordConstructor) {
			write((Expr.RecordConstructor) expr);
		} else if(expr instanceof Expr.Unary) {
			write((Expr.Unary) expr);
		} else if(expr instanceof Expr.Variable) {
			write((Expr.Variable) expr);
		} else {
			internalFailure("unknown expression encountered (" + expr + ")", file.filename,expr);
		}
	}

	public void write(Expr.Binary expr) {

		//Must convert a range expression
		if (expr.getOp() == Expr.BOp.RANGE) {
			out.print("($_range_$(");
			write(expr.getLhs());
			out.print(".num, ");
			write(expr.getRhs());
			out.print(".num))");
			return;
		}

		//Have to handle the case where
		//Working with numbers - must call the method on the JavaScript object
		switch (expr.getOp()) {

		case AND:
		case OR:
		case APPEND:
			out.print("(");
			write(expr.getLhs());
			break;

		case ADD:
			out.print("(");
			write(expr.getLhs());
			out.print(".add(");
			write(expr.getRhs());
			out.print("))");
			return;

		case DIV:
			out.print("(");
			write(expr.getLhs());
			out.print(".div(");
			write(expr.getRhs());
			out.print("))");
			return;

		case MUL:
			out.print("(");
			write(expr.getLhs());
			out.print("(");
			write(expr.getLhs());
			out.print(".mul(");
			write(expr.getRhs());
			out.print("))");
			return;

		case REM:
			out.print("(");
			write(expr.getLhs());
			out.print(".rem(");
			write(expr.getRhs());
			out.print("))");
			return;

		case SUB:
			out.print("(");
			write(expr.getLhs());
			out.print(".sub(");
			write(expr.getRhs());
			out.print("))");
			return;

		case GTEQ:
		case GT:
			out.print("($_gt_$(");
			write(expr.getLhs());
			out.print(", ");
			write(expr.getRhs());
			out.print(", ");
			if (expr.getOp() == Expr.BOp.GT)
				out.print(" false))");
			else
				out.print(" true))");
			return;

		case LT:
		case LTEQ:
			out.print("($_lt_$(");
			write(expr.getLhs());
			out.print(", ");
			write(expr.getRhs());
			out.print(", ");
			if (expr.getOp() == Expr.BOp.LT)
				out.print(" false))");
			else
				out.print(" true))");
			return;

		case NEQ:
			out.print("($_equals_$(");
			write(expr.getLhs());
			out.print(", ");
			write(expr.getRhs());
			out.print(", false))");
			return;
		case EQ:
			out.print("($_equals_$(");
			write(expr.getLhs());
			out.print(", ");
			write(expr.getRhs());
			out.print(", true))");
			return;
		}

		out.print(" " + expr.getOp() + " ");
		write(expr.getRhs());
		out.print(")");
	}

	public void write(Expr.Cast expr) {

		//We need to check for the case where casting one of the number types
		//to the other number type, as this affects how we print the number
		Type t = expr.getType();
		if (t instanceof Type.Record) {
			castRecord(expr.getSource(), (Type.Record) t);
		}
		else if (t instanceof Type.List) {
			castList(expr.getSource(), (Type.List)t);
		}
		else if (t instanceof Type.Real) {
			out.print("(new $_Float_$(");
			write(expr.getSource());
			out.print("))");
			return;
		}
		else if (t instanceof Type.Int) {
			out.print("(new $_Integer_$(");
			write(expr.getSource());
			out.print("))");
			return;
		}
		else write(expr.getSource());
	}
	
	public void write(Expr.Constant expr) {

		Type t = expr.attribute(Attribute.Type.class).type;
		Object val = expr.getValue();

		if (t instanceof Type.Real) {
			out.print("new $_Float_$(");
			out.print(val + ")");
		}
		else if (t instanceof Type.Int) {
			out.print("new $_Integer_$(");
			out.print(val + ")");
		}
		else if (val instanceof StringBuffer) {
			String s = ((StringBuffer) val).toString();
			out.print("\"");
			out.print(s);
			out.print("\"");
		} else
			out.print(val);
	}

	public void write(Expr.IndexOf expr) {
		write(expr.getSource());
		out.print("[");
		write(expr.getIndex());
		out.print(".num]");
	}

	public void write(Expr.Invoke expr) {
		out.print(expr.getName() + "(");
		boolean firstTime=true;
		for(Expr arg : expr.getArguments()) {
			if(!firstTime) {
				out.print(",");
			}
			firstTime=false;
			write(arg);
		}
		out.print(")");
	}

	public void write(Expr.ListConstructor expr) {
		out.print("[");
		boolean firstTime=true;
		for(Expr arg : expr.getArguments()) {
			if(!firstTime) {
				out.print(",");
			}
			firstTime=false;
			write(arg);
		}
		out.print("]");
	}

	public void write(Expr.RecordAccess expr) {
		write(expr.getSource());
		out.print("." + expr.getName());
	}

	public void write(Expr.RecordConstructor expr) {
		out.print("({");
		boolean firstTime=true;
		for(Pair<String,Expr> p : expr.getFields()) {
			if(!firstTime) {
				out.print(",");
			}
			firstTime=false;
			out.print(p.first() + ": ");
			write(p.second());
		}
		out.print("})");
	}

	public void write(Expr.Unary expr) {
		out.print("(");
		switch(expr.getOp()) {
		case NOT:
			out.print(expr.getOp());
			write(expr.getExpr());
			break;
		case NEG:
			Type t = expr.getExpr().attribute(Attribute.Type.class).type;
			if (t instanceof Type.Int)
				out.print("new $_Integer_$(");
			else
				out.print("new $_Float_$(");
			out.print(expr.getOp() + "(");
			write(expr.getExpr());
			out.print(".num)");
			break;
		case LENGTHOF:
			write(expr.getExpr());
			out.print(".length");
		}
		out.print(")");
	}

	public void write(Expr.Variable expr) {
		out.print(expr.getName());
	}

	public void indent(int indent) {
		for(int i=0;i!=indent;++i) {
			out.print("    ");
		}
	}
}
