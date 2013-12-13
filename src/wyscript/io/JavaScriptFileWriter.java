package wyscript.io;

import static wyscript.util.SyntaxError.internalFailure;

import java.io.*;
import java.util.List;

import wyscript.lang.*;
import wyscript.util.*;

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

		//Need to create additional helper functions
		setupFunctions();

		for(WyscriptFile.Decl declaration : wf.declarations) {
			if(declaration instanceof WyscriptFile.FunDecl) {
				write((WyscriptFile.FunDecl) declaration);
			}
		}
	}

	/**
	 * Sets up any additional functions necessary to properly
	 * convert from WyScript to JavaScript (such as a special print for lists)
	 */
	private void setupFunctions() {

		//printList method necessary to print WyScript's List type
		StringBuilder sb = new StringBuilder();
		sb.append("function printList(list) {\n");
		sb.append("sysout.print(\"[\");\n");
		sb.append("var first = true;\n");
		sb.append("for (var i = 0; i < list.length; i++) {\n");
		sb.append("if(!first)\n");
		sb.append("sysout.print(\", \");\n");
		sb.append("var elem = list[i];\n");
		sb.append("first = false;\n");
		sb.append("if (elem instanceof list)\n");
		sb.append("printList(elem);\n");
		sb.append("else\n");
		sb.append("sysout.print(elem);\n");
		sb.append("}\n");
		sb.append("sysout.print(\"]\\n\");\n");
		sb.append("}\n");

		out.print(sb.toString());
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
		} else {
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

		//Handle printing a list
		if (type instanceof Type.List) {
			out.print("printList(");
			write(stmt.getExpr());
			out.print(")");
			return;
		}

		out.print("sysout.println(");

		//Ints need to be truncated
		if(type instanceof Type.Int) {
			write(stmt.getExpr());
			out.print(".toFixed()");
		}

		else {
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
			write((Expr.ListConstructor) expr);
		} else if(expr instanceof Expr.Unary) {
			write((Expr.Unary) expr);
		} else if(expr instanceof Expr.Variable) {
			write((Expr.Variable) expr);
		} else {
			internalFailure("unknown expression encountered (" + expr + ")", file.filename,expr);
		}
	}

	public void write(Expr.Binary expr) {
		out.print("(");
		write(expr.getLhs());
		out.print(" " + expr.getOp() + " ");
		write(expr.getRhs());
		out.print(")");
	}

	public void write(Expr.Cast expr) {
		write(expr.getSource());
	}

	public void write(Expr.Constant expr) {
		Object val = expr.getValue();
		if (val instanceof StringBuffer) {
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
		out.print("]");
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
		out.print("{");
		boolean firstTime=true;
		for(Pair<String,Expr> p : expr.getFields()) {
			if(!firstTime) {
				out.print(",");
			}
			firstTime=false;
			out.print(p.first() + ": ");
			write(p.second());
		}
		out.print("}");
	}

	public void write(Expr.Unary expr) {
		out.print("(");
		switch(expr.getOp()) {
		case NOT:
		case NEG:
			out.print(expr.getOp());
			write(expr.getExpr());
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
