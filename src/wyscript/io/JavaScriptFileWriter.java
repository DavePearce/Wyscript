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
	private HashMap<String, Type> userTypes;

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

	/**
	 * Sets up any additional functions necessary to properly
	 * convert from WyScript to JavaScript
	 */
	private void setupFunctions() {

		setUpPrintFunctions();
		setUpCastFunctions();
		setUpNumberFunctions();
		setUpBinaryFunctions();
	}

	private void setUpBinaryFunctions() {
		//First, the range function
		StringBuilder sb = new StringBuilder("function $_range_$(lower, upper) {\n");
		sb.append("var $_result_$ = [];\n");
		sb.append("var $_count_$ = 0;\n");
		sb.append("for (var $_tmp_$ = lower; $_tmp_$ < upper; $_tmp_$++) {\n");
		sb.append("$_result_$[$_count_$] = new $_Integer_$($_tmp_$);\n");
		sb.append("$_count_$++;\n}\n");
		sb.append("return $_result_$;\n}\n");

		out.print(sb.toString());

		//Next, the 3 comparison functions - =/!=, >/>= and </<=
		sb = new StringBuilder("function $_equals_$(lhs, rhs, isEqual) {\n");
		sb.append("var left = lhs;\n");
		sb.append("if (typeof left.num !== 'undefined') left = left.num;\n");
		sb.append("var right = rhs;\n");
		sb.append("if (typeof right.num !== 'undefined') right = right.num;\n");
		sb.append("if (isEqual) return (left === right);\n");
		sb.append("else return (left !== right);\n}\n");

		out.print(sb.toString());

		sb = new StringBuilder("function $_lt_$(lhs, rhs, isEqual) {\n");
		sb.append("var left = lhs;\n");
		sb.append("if (typeof left.num !== 'undefined') left = left.num;\n");
		sb.append("var right = rhs;\n");
		sb.append("if (typeof right.num !== 'undefined') right = right.num;\n");
		sb.append("if (isEqual) return (left <= right);\n");
		sb.append("else return (left < right);\n}\n");

		out.print(sb.toString());

		sb = new StringBuilder("function $_gt_$(lhs, rhs, isEqual) {\n");
		sb.append("var left = lhs;\n");
		sb.append("if (typeof left.num !== 'undefined') left = left.num;\n");
		sb.append("var right = rhs;\n");
		sb.append("if (typeof right.num !== 'undefined') right = right.num;\n");
		sb.append("if (isEqual) return (left >= right);\n");
		sb.append("else return (left > right);\n}\n");

		out.print(sb.toString());
	}

	private void setUpPrintFunctions() {

		// First off, create a general print method for the more specific
		// methods to reference
		StringBuilder sb = new StringBuilder();
		sb.append("function $_print_$(obj, endLine) {\n");
		sb.append("var end = (endLine) ? \"\\n\" : \"\";\n");
		sb.append("if (obj instanceof Array)\n");
		sb.append("$_printList_$(obj, endLine);\n");
		sb.append("else if(obj instanceof $_Float_$) {\n");
		sb.append("if (endLine)\n");
		sb.append("sysout.println(obj.num);\n");
		sb.append("else sysout.print(obj.num);\n}\n");
		sb.append("else if(obj instanceof $_Integer_$)\n");
		sb.append("sysout.print(obj.num.toFixed() + end);\n");
		sb.append("else if(obj instanceof Object)\n");
		sb.append("$_printRecord_$(obj, endLine);\n");
		sb.append("else if(typeof obj === 'number')\n");
		sb.append("$_printNumber_$(obj, endLine);\n");
		sb.append("else sysout.print(obj + end);\n");
		sb.append("}\n");

		out.print(sb.toString());

		//Simple number printer
		sb = new StringBuilder("function $_printNumber_$(obj, endLine) {\n");
		sb.append("var end = (endLine) ? \"\\n\" : \"\";\n");
		sb.append("if (String(obj)[String(obj).length-2] !== '.')\n");
		sb.append("sysout.print(obj.toFixed() + end);\n");
		sb.append("else {\n");
		sb.append("if (endLine)\n");
		sb.append("sysout.println(obj);\n");
		sb.append("else sysout.print(obj);\n}\n}\n");

		out.print(sb.toString());

		// $_printList_$ method necessary to print WyScript's List type
		sb = new StringBuilder();
		sb.append("function $_printList_$(list, endLine) {\n");
		sb.append("var end = (endLine) ? \"\\n\" : \"\";\n");
		sb.append("sysout.print(\"[\");\n");
		sb.append("var first = true;\n");
		sb.append("for (var i = 0; i < list.length; i++) {\n");
		sb.append("if(!first)\n");
		sb.append("sysout.print(\", \");\n");
		sb.append("var elem = list[i];\n");
		sb.append("first = false;\n");
		sb.append("$_print_$(elem, false);\n");
		sb.append("}\n");
		sb.append("sysout.print(\"]\" + end);\n");
		sb.append("}\n");

		out.print(sb.toString());

		// Now the $_printRecord_$ method
		sb = new StringBuilder("function $_printRecord_$(record, endLine) {\n");
		sb.append("var end = (endLine) ? \"\\n\" : \"\";\n");
		sb.append("sysout.print(\"{\");\n");
		sb.append("var first = true;\n");
		sb.append("var keys = Object.keys(record);\n");
		sb.append("keys.sort();\n");
		sb.append("for (var i = 0; i < keys.length; i++) {\n");
		sb.append("var key = keys[i];\n");
		sb.append("if (!first)\n");
		sb.append("sysout.print(\",\");\n");
		sb.append("first = false;\n");
		sb.append("sysout.print(key + \":\");\n");
		sb.append("var value = record[key];\n");
		sb.append("$_print_$(value, false);\n");
		sb.append("}\n");
		sb.append("sysout.print(\"}\" + end);\n");
		sb.append("}\n");

		out.print(sb.toString());

	}

	private void setUpCastFunctions() {

		//Set up list-casting method
		StringBuilder sb = new StringBuilder("function $_castList_$(list, typeList) {\n");
		sb.append("for (var i = 0; i < list.length; i++) {\n");
		sb.append("if (list[i] instanceof Array)\n");
		sb.append("list[i] = $_castList_$(list[i]);\n");
		sb.append("else if (!(list[i] instanceof $_Float_$ || list[i] instanceof $_Integer_$)");
		sb.append(" && list[i] instanceof Object)\n");
		sb.append("list[i] = $_castRecord_$(list[i], typeList[0], typeList.slice(1));\n");
		sb.append("else list[i] = list[i].cast();\n}\n");
		sb.append("return list;\n}\n");

		out.print(sb.toString());

		sb = new StringBuilder("function $_castRecord_$(record, name, list) {\n");
		sb.append("if (list.length > 0)\n");
		sb.append("record[list[0]] = $_castRecord_$(record[list[0]], name, list.slice(1));\n");
		sb.append("else record[name] = record[name].cast();\n");
		sb.append("return record;\n}\n");

		out.print(sb.toString());
	}

	private void setUpNumberFunctions() {

		// Create a Float function to represent a WyScript Real
		StringBuilder sb = new StringBuilder("function $_Float_$(i) {\n");
		sb.append("if (typeof i.num !== 'undefined') this.num = i.num;\n");
		sb.append("else this.num = i;\n}\n\n");
		sb.append("$_Float_$.prototype.add = function(other) {\n");
		sb.append("return new $_Float_$(this.num + other.num);\n}\n");
		sb.append("$_Float_$.prototype.sub = function(other) {\n");
		sb.append("return new $_Float_$(this.num - other.num);\n}\n");
		sb.append("$_Float_$.prototype.mul = function(other) {\n");
		sb.append("return new $_Float_$(this.num * other.num);\n}\n");
		sb.append("$_Float_$.prototype.div = function(other) {\n");
		sb.append("return new $_Float_$(this.num / other.num);\n}\n");
		sb.append("$_Float_$.prototype.rem = function(other) {\n");
		sb.append("return new $_Float_$(this.num % other.num);\n}\n");
		sb.append("$_Float_$.prototype.cast = function() {\n");
		sb.append("return new $_Integer_$(this.num);\n}\n");

		out.print(sb.toString());

		// And an Integer function for a WyScript int
		sb = new StringBuilder("function $_Integer_$(i) {\n");
		sb.append("if (typeof i.num !== 'undefined') this.num = ~~(i.num);\n");
		sb.append("else this.num = ~~i;\n}\n\n");
		sb.append("$_Integer_$.prototype.add = function(other) {\n");
		sb.append("if (other instanceof $_Integer_$)\n");
		sb.append("return new $_Integer_$(this.num + other.num);\n");
		sb.append("else return new $_Float_$(this.num + other.num);\n}\n");
		sb.append("$_Integer_$.prototype.sub = function(other) {\n");
		sb.append("if (other instanceof $_Integer_$)\n");
		sb.append("return new $_Integer_$(this.num - other.num);\n");
		sb.append("else return new $_Float_$(this.num - other.num);\n}\n");
		sb.append("$_Integer_$.prototype.mul = function(other) {\n");
		sb.append("if (other instanceof $_Integer_$)\n");
		sb.append("return new $_Integer_$(this.num * other.num);\n");
		sb.append("else return new $_Float_$(this.num * other.num);\n}\n");
		sb.append("$_Integer_$.prototype.div = function(other) {\n");
		sb.append("var tmp = this.num / other.num;\n");
		sb.append("if (other instanceof $_Integer_$) {\n");
		sb.append("return new $_Integer_$(~~tmp);\n}\n");
		sb.append("else return new $_Float_$(this.num / other.num);\n}\n");
		sb.append("$_Integer_$.prototype.rem = function(other) {\n");
		sb.append("if (other instanceof $_Integer_$)\n");
		sb.append("return new $_Integer_$(this.num % other.num);\n");
		sb.append("else return new $_Float_$(this.num % other.num);\n}\n");
		sb.append("$_Integer_$.prototype.cast = function() {\n");
		sb.append("return new $_Float_$(this.num);\n}\n");

		out.print(sb.toString());
	}

	public void write(WyscriptFile.ConstDecl cd) {
		Type t = cd.constant.attribute(Attribute.Type.class).type;
		out.print("var "+ cd.name() + " = ");

		if (t instanceof Type.Real) {
			out.print("new $_Float_$(");
			write(cd.constant);
			out.print(")");
		}
		else if (t instanceof Type.Int) {
			out.print("new $_Integer_$(");
			write(cd.constant);
			out.print(")");
		}
		else
			write(cd.constant);
		out.println(";");
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

	/**
	 * Writes a for each loop. Only consideration is
	 * if dealing with a range, must convert that to a list first
	 *
	 */
	public void write(Stmt.For stmt, int indent) {

		//After being dealt with by write, $$__tmp__$$ will be a list
		indent(indent);
		out.print("var $$__tmp__$$ = ");
		write(stmt.getSource());
		out.println(";");
		indent(indent);

		//Simulate a foreach loop by iterating over the list, and defining the index value to be equal
		//to the element at the current index
		out.print("for(var $_tmp_$ = 0; $_tmp_$ < $$__tmp__$$.length; $_tmp_$++) {\n");
		indent(indent+1);
		out.println("var " + stmt.getIndex().getName() + " = $$__tmp__$$[$_tmp_$];");
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
		out.print("$_print_$(");
		write(stmt.getExpr());
		out.print(", true)");
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

	/**
	 * Helper method used to determine whether or not we
	 * need to take extra care casting a list - as is the
	 * case when the list contains a number type to cast.
	 */
	private void castList(Expr source, Type.List type) {

		Type t = type.getElement();
		Type actual = source.attribute(Attribute.Type.class).type;

		while (t instanceof Type.List) {
			t = ((Type.List)t).getElement();
			actual = ((Type.List)actual).getElement();
		}
		List<String> typeList = new ArrayList<String>();

		if (t instanceof Type.Record && !t.equals(actual))
			typeList = getRecordTypeList((Type.Record)actual, (Type.Record)t);

		if (!t.equals(actual)) {
			out.print("($_castList_$(");
			write(source);
			out.print(",");
			StringBuilder list = new StringBuilder("[");
			boolean first = true;
			for (String s : typeList) {
				if (!first)
					list.append(", ");
				first = false;
				list.append(s);
			}
			list.append("]");
			out.print(list.toString() + "))");
		}
		else write(source);
	}

	private List<String> getRecordTypeList(Type.Record actual, Type.Record type) {
		Type.Record current = type;
		String castedName = "";
		Type casted = null;
		List<String> fieldList = new ArrayList<String>();

		while (casted == null) {
			for (String s : actual.getFields().keySet()) {
				if (actual.getFields().get(s).equals(current.getFields().get(s)))
					continue;
				else {
					Type t = current.getFields().get(s);
					if (t instanceof Type.Record) {
						fieldList.add(s);
						current = (Type.Record)t;
						actual = (Type.Record)actual.getFields().get(s);
					}
					else {
						casted = t;
						castedName = s;
					}
				}
			}
			break; //Signals a 'pointless' cast
		}
		if (casted == null)
			return null; //pointless cast, no difference in types
		else {
			fieldList.add(0, castedName);
			return fieldList;
		}
	}

	private void castRecord(Expr source, Type.Record type) {
		Type src = source.attribute(Attribute.Type.class).type;
		if (src instanceof Type.Named)
			src = userTypes.get(((Type.Named)src).getName());
		Type.Record actual = (Type.Record) src;
		Type.Record current = type;
		String castedName = "";
		Type casted = null;
		List<String> fieldList = new ArrayList<String>();

		while (casted == null) {
			for (String s : actual.getFields().keySet()) {
				if (actual.getFields().get(s).equals(current.getFields().get(s)))
					continue;
				else {
					Type t = current.getFields().get(s);
					if (t instanceof Type.Record) {
						fieldList.add(s);
						current = (Type.Record)t;
						actual = (Type.Record)actual.getFields().get(s);
					}
					else {
						casted = t;
						castedName = s;
					}
				}
			}
			break; //Signals a 'pointless' cast
		}
		if (casted == null) {
			write(source);
		}
		else {
			if (casted instanceof Type.Int || casted instanceof Type.Real) {
				out.print("($_castRecord_$(");
				write(source);
				StringBuilder list = new StringBuilder("[");
				boolean first = true;
				for (String s : fieldList) {
					if (!first)
						list.append(", ");
					first = false;
					list.append(s);
				}
				list.append("]");
				out.print(", \""+ castedName + "\", " + list.toString() + "))");
			}
			else if (casted instanceof Type.List) {
				while (casted instanceof Type.List) {
					casted = ((Type.List)casted).getElement();
				}
				if (casted instanceof Type.Int || casted instanceof Type.Real) {
					out.print("($_castRecord_$(");
					write(source);
					out.print(",");
					StringBuilder list = new StringBuilder("[");
					boolean first = true;
					for (String s : fieldList) {
						if (!first)
							list.append(", ");
						first = false;
						list.append(s);
					}
					list.append("]");
					out.print(", "+ castedName + ", " + list.toString() + "))");
				}
				else
					write(source);
			}
			else
				write(source);
		}
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
