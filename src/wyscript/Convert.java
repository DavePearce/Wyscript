// This file is part of the WyScript Compiler (wysc).
//
// The WyScript Compiler is free software; you can redistribute
// it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// The WyScript Compiler is distributed in the hope that it
// will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public
// License along with the WyScript Compiler. If not, see
// <http://www.gnu.org/licenses/>
//
// Copyright 2013, David James Pearce.

package wyscript;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.Map;

import wyscript.io.*;
import wyscript.lang.*;
import wyscript.util.*;

public class Convert {

	public static PrintStream errout;

	static {
		try {
			errout = new PrintStream(System.err, true, "UTF8");
		} catch (Exception e) {
			errout = System.err;
		}
	}

	private static enum Mode { interpret, js };
	
	public static boolean run(String[] args) {
		boolean verbose = false;
		int fileArgsBegin = 0;
		Mode mode = Mode.interpret;
		
		for (int i = 0; i != args.length; ++i) {
			if (args[i].startsWith("-")) {
				String arg = args[i];
				if (arg.equals("-help")) {
					usage();
					System.exit(0);
				} else if (arg.equals("-version")) {
					System.out.println("While Language Compiler (wysc)");
					System.exit(0);
				} else if (arg.equals("-verbose")) {
					verbose = true;
				} else if (arg.equals("-js")) {
					mode = Mode.js;
				} else {
					throw new RuntimeException("Unknown option: " + args[i]);
				}

				fileArgsBegin = i + 1;
			}
		}

		if (fileArgsBegin == args.length) {
			usage();
			return false;
		}

		try {
			String filename = args[fileArgsBegin];
			File srcFile = new File(filename);

			// First, lex and parse the source file
			OriginalLexer lexer = new OriginalLexer(srcFile.getPath());
			OriginalParser parser = new OriginalParser(srcFile.getPath(), lexer.scan());
			WyscriptFile ast = parser.read();
			convert(ast);
			
		} catch (SyntaxError e) {
			if (e.filename() != null) {
				e.outputSourceError(System.out);
			} else {
				System.err.println("syntax error (" + e.getMessage() + ").");
			}

			if (verbose) {
				e.printStackTrace(errout);
			}

			return false;
		} catch (Exception e) {
			errout.println("Error: " + e.getMessage());
			if (verbose) {
				e.printStackTrace(errout);
			}
			return false;
		}

		return true;
	}

	public static void main(String[] args) throws Exception {
		run(args);
	}

	/**
	 * Print out information regarding command-line arguments
	 * 
	 */
	public static void usage() {
		String[][] info = {
				{ "version", "Print version information" },
				{ "verbose",
						"Print detailed information on what the compiler is doing" } };

		System.out.println("usage: wyjs <options> <source-files>");
		System.out.println("Options:");

		// first, work out gap information
		int gap = 0;

		for (String[] p : info) {
			gap = Math.max(gap, p[0].length() + 5);
		}

		// now, print the information
		for (String[] p : info) {
			System.out.print("  -" + p[0]);
			int rest = gap - p[0].length();
			for (int i = 0; i != rest; ++i) {
				System.out.print(" ");
			}
			System.out.println(p[1]);
		}
	}
	
	private static void convert(WyscriptFile file) {
		for(WyscriptFile.Decl decl : file.declarations) {
			if(decl instanceof WyscriptFile.ConstDecl) {
				print((WyscriptFile.ConstDecl) decl);	
			} else if(decl instanceof WyscriptFile.TypeDecl) {
				print((WyscriptFile.TypeDecl) decl);
			} else {
				print((WyscriptFile.FunDecl) decl);
			}		
		}
	}
	
	public static void print(WyscriptFile.TypeDecl decl) {
		System.out.print("type " + decl.name + " is ");
		print(decl.type);
		System.out.println("\n");
	}
	
	public static void print(WyscriptFile.ConstDecl decl) {
		System.out.print("constant " + decl.name + " is ");
		print(decl.constant);
		System.out.println("\n");
	}
	
	public static void print(WyscriptFile.FunDecl decl) {
		print(decl.ret);
		System.out.print(" " + decl.name + "(");
		boolean firstTime = true;
		for(WyscriptFile.Parameter p : decl.parameters) {
			if(!firstTime) {
				System.out.print(", ");
			}
			firstTime=false;
			print(p.type);
			System.out.print(" " + p.name);
		}
		System.out.println("):");
		print(decl.statements,4);
		System.out.println();
	}
	
	public static void print(List<Stmt> stmts, int indent) {
		for(Stmt stmt : stmts) {
			if(stmt instanceof Stmt.Print) {
				print((Stmt.Print) stmt,indent);
			} else if(stmt instanceof Stmt.Return) {
				print((Stmt.Return) stmt,indent);
			} else if(stmt instanceof Stmt.Assign) {
				print((Stmt.Assign) stmt,indent);
			} else if(stmt instanceof Stmt.For) {
				print((Stmt.For) stmt,indent);
			} else if(stmt instanceof Stmt.IfElse) {
				print((Stmt.IfElse) stmt,indent);
			} else if(stmt instanceof Stmt.While) {
				print((Stmt.While) stmt,indent);
			} else if(stmt instanceof Stmt.VariableDeclaration) {
				print((Stmt.VariableDeclaration) stmt,indent);
			} else if(stmt instanceof Expr.Invoke) {
				indent(indent);
				print((Expr.Invoke) stmt);
			} else {
				throw new RuntimeException("Unknown statement: " + stmt.getClass().getName());
			}
		}
	}
	
	public static void print(Stmt.Assign stmt, int indent) {
		indent(indent);
		print(stmt.getLhs());
		System.out.print(" = ");
		print(stmt.getRhs());
		System.out.println();
	}
	
	public static void print(Stmt.For stmt, int indent) {
		// TODO
	}
	
	public static void print(Stmt.IfElse stmt, int indent) {
		indent(indent);
		System.out.print("if ");
		print(stmt.getCondition());
		System.out.println(":");
		print(stmt.getTrueBranch(),indent+4);
		if(!stmt.getFalseBranch().isEmpty()) {
			indent(indent);System.out.println("else:");
			print(stmt.getFalseBranch(),indent+4);
		}
	}
	
	public static void print(Stmt.While stmt, int indent) {
		indent(indent);
		System.out.print("while ");
		print(stmt.getCondition());
		System.out.println(":");
		print(stmt.getBody(),indent+4);
	}
	
	public static void print(Stmt.VariableDeclaration stmt, int indent) {
		indent(indent);
		print(stmt.getType());
		System.out.print(" " + stmt.getName());
		if(stmt.getExpr() != null) {
			System.out.print(" = ");
			print(stmt.getExpr());			
		}
		System.out.println();
	}
	
	public static void print(Stmt.Print stmt, int indent) {
		indent(indent);
		System.out.print("print ");
		print(stmt.getExpr());
		System.out.println();
	}
	
	public static void print(Stmt.Return stmt, int indent) {
		indent(indent);
		System.out.print("return");
		if(stmt.getExpr() != null) {
			System.out.print(" ");
			print(stmt.getExpr());
		}
		System.out.println();
	}
	
	public static void print(Type t) {
		if(t instanceof Type.Void) {
			System.out.print("void");
		} else if(t instanceof Type.Null) {
			System.out.print("null");
		} else if(t instanceof Type.Bool) {
			System.out.print("bool");
		} else if(t instanceof Type.Char) {
			System.out.print("char");
		} else if(t instanceof Type.Int) {
			System.out.print("int");
		} else if(t instanceof Type.Real) {
			System.out.print("real");
		} else if(t instanceof Type.Strung) {
			System.out.print("string");
		} else if(t instanceof Type.Named) {
			Type.Named l = (Type.Named) t;
			System.out.print(l.getName());
		} else if(t instanceof Type.List) {
			Type.List l = (Type.List) t;
			System.out.print("[");
			print(l.getElement());
			System.out.print("]");
		} else if(t instanceof Type.Record) {
			Type.Record l = (Type.Record) t;
			System.out.print("{");
			boolean firstTime=true;
			for(Map.Entry<String,Type> ce : l.getFields().entrySet()) {
				if(!firstTime) {
					System.out.print(", ");
				}
				firstTime=false;
				print(ce.getValue());
				System.out.print(" " + ce.getKey());				
			}
			System.out.print("}");
		} else if(t instanceof Type.Union) {
			Type.Union u = (Type.Union) t;
			boolean firstTime=true;
			for(Type b : u.getBounds()) {
				if(!firstTime) {
					System.out.print("|");
				}
				firstTime=false;
				print(b);
			}
		} else {
			throw new RuntimeException("Unknown type: " + t.getClass().getName());
		}
	}
	
	public static void print(Expr e) {
		if(e instanceof Expr.Constant) {
			Expr.Constant c = (Expr.Constant) e;
			Object constant = c.getValue();
			if(constant instanceof String) {
				System.out.print("\"" + constant + "\"");
			} else if(constant instanceof Character) {
				System.out.print("\'" + constant + "\'");
			} else {
				System.out.print(c.getValue());
			}
		} else if(e instanceof Expr.Variable) {
			Expr.Variable v = (Expr.Variable) e;
			System.out.print(v.getName());
		} else if(e instanceof Expr.Unary) {
			Expr.Unary u = (Expr.Unary) e;
			switch(u.getOp()) {
			case NOT:				
			case NEG:
				System.out.print(u.getOp() + " ");
				print(u.getExpr());				
				break;
			case LENGTHOF:
				System.out.print("|");
				print(u.getExpr());
				System.out.print("|");
			}
		} else if(e instanceof Expr.Binary) {
			Expr.Binary b = (Expr.Binary) e;
			print(b.getLhs());
			System.out.print(" " + b.getOp() + " ");
			print(b.getRhs());
		} else if(e instanceof Expr.Cast) {
			Expr.Cast c = (Expr.Cast) e;
			System.out.print("(");
			print(c.getType());
			System.out.print(") ");
			print(c.getSource());
		} else if(e instanceof Expr.IndexOf) {
			Expr.IndexOf c = (Expr.IndexOf) e;
			print(c.getSource());
			System.out.print("[");
			print(c.getIndex());
			System.out.print("]");
		} else if(e instanceof Expr.Invoke) {
			Expr.Invoke c = (Expr.Invoke) e;
			System.out.print(c.getName() + "(");
			boolean firstTime=true;
			for(Expr ce : c.getArguments()) {
				if(!firstTime) {
					System.out.print(", ");
				}
				firstTime=false;
				print(ce);
			}
			System.out.print(")");
		} else if(e instanceof Expr.ListConstructor) {
			Expr.ListConstructor c = (Expr.ListConstructor) e;
			System.out.print("[");
			boolean firstTime=true;
			for(Expr ce : c.getArguments()) {
				if(!firstTime) {
					System.out.print(", ");
				}
				firstTime=false;
				print(ce);
			}
			System.out.print("]");
		} else if(e instanceof Expr.RecordConstructor) {
			Expr.RecordConstructor c = (Expr.RecordConstructor) e;
			System.out.print("{");
			boolean firstTime=true;
			for(Pair<String,Expr> ce : c.getFields()) {
				if(!firstTime) {
					System.out.print(", ");
				}
				firstTime=false;
				System.out.print(ce.first() + ": ");
				print(ce.second());
			}
			System.out.print("}");
		} else if(e instanceof Expr.RecordAccess) {
			Expr.RecordAccess c = (Expr.RecordAccess) e;
			print(c.getSource());
			System.out.print("." + c.getName());
		} else {
			throw new RuntimeException("Unknown expression: " + e.getClass().getName());
		}
	}
	
	public static void indent(int indent) {
		for(int i=0;i!=indent;++i) {
			System.out.print(" ");
		}
	}
}
