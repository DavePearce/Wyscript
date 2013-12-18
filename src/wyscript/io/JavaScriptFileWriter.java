package wyscript.io;

import static wyscript.util.SyntaxError.internalFailure;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


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

		setUpTypes();
		setUpNumberFunctions();
		setUpBinaryFunctions();

		//Finally, the misc functions for overlap between strings and lists
		// eg. the string replace/mutate function, length and indexOf functions
		out.println("function $_stringIndexReplace_$(str, index, c) {");
		indent(1);
		out.println("var num = index;");
		indent(1);
		out.println("if (typeof index.num !== 'undefined') num = index.num;");
		indent(1);
		out.println("var tmp = str.split('');");
		indent(1);
		out.println("tmp[num] = c;");
		indent(1);
		out.println("return tmp.join('');}");

		out.println("function $_indexOf_$(obj, index) {");
		out.println("if (obj instanceof String || typeof obj === 'string')");
		out.println("return obj.charAt(index);");
		out.println("else return obj.getValue(index);\n}");

		out.println("function $_length_$(obj) {");
		out.println("if (obj instanceof String || typeof obj === 'string')");
		out.println("return new $_Integer_$(obj.length);");
		out.println("else return obj.length();\n}\n");

	}

	private void setUpTypes() {
		//First, set up the record type
		out.println("function $_Record_$(listNames, listValues) {");
		indent(1);
		out.println("this.names = listNames;");
		indent(1);
		out.println("this.values = listValues;\n}\n");

		out.println("$_Record_$.prototype.getValue = function(name) {");
		indent(1);
		out.println("var index = this.names.indexOf(name);");
		indent(1);
		out.println("if (index === -1 || index >= this.values.length)");
		indent(2);
		out.println("return;");
		indent(1);
		out.println("else return this.values[index];\n}");

		out.println("$_Record_$.prototype.hasKey = function(name) {");
		indent(1);
		out.println("return (this.names.findIndex(name) !== -1);\n}");

		out.println("$_Record_$.prototype.setValue = function(name, key) {");
		indent(1);
		out.println("var index = this.names.indexOf(name);");
		indent(1);
		out.println("if (index === -1 || index >= this.values.length)");
		indent(2);
		out.println("return;");
		indent(1);
		out.println("else this.values[index] = key;\n}");

		out.println("$_Record_$.prototype.cast = function(name, fieldList) {");
		out.println("var result = this.clone();");
		out.println("if (fieldList.length > 0) {");
		out.println("var index = this.names.indexOf(fieldList[0]);");
		out.println("result.values[index] = this.values[index].cast(name, fieldList.slice[1]);");
		out.println("}");
		out.println("else {");
		out.println("var index = this.names.indexOf(name);");
		out.println("result.values[index] = this.values[index].cast();");
		out.println("return result;");
		out.println("}\n}");

		out.println("$_Record_$.prototype.clone = function() {");
		indent(1);
		out.println("var cnames = [];");
		indent(1);
		out.println("var cvalues = [];");
		indent(1);
		out.println("for (var i = 0; i < this.names.length; i++) {");
		indent(2);
		out.println("cnames[i] = this.names[i];");
		indent(2);
		out.println("var elem = this.values[i];");
		indent(2);
		out.println("if (elem instanceof $_List_$ || elem instanceof $_Record_$)");
		indent(3);
		out.println("elem = elem.clone();");
		indent(2);
		out.println("cvalues[i] = elem;");
		indent(1);
		out.println("}");
		indent(1);
		out.println("return new $_Record_$(cnames, cvalues);\n}");

		out.println("$_Record_$.prototype.toString = function() {");
		indent(1);
		out.println("var str = '{';");
		indent(1);
		out.println("var tmpNames = []");
		indent(1);
		out.println("for (var i = 0; i < this.names.length; i++)");
		indent(2);
		out.println("tmpNames[i] = this.names[i];");
		indent(1);
		out.println("tmpNames.sort();");
		indent(1);
		out.println("var first = true;");
		indent(1);
		out.println("for (var i = 0; i < this.names.length; i++) {");
		indent(2);
		out.println("if (!first)");
		indent(3);
		out.println("str += ',';");
		indent(2);
		out.println("first = false;");
		indent(2);
		out.println("str += tmpNames[i];");
		indent(2);
		out.println("str += ':';");
		indent(2);
		out.println("var val = this.values[this.names.indexOf(tmpNames[i])];");
		indent(2);
		out.println("str += val;");
		indent(1);
		out.println("}");
		indent(1);
		out.println("str += '}';");
		indent(1);
		out.println("return str;\n}\n");

		//And now make the list function
		out.println("function $_List_$(list, type) {");
		indent(1);
		out.println("this.list = list;\n}");

		out.println("$_List_$.prototype.getValue = function(index) {");
		indent(1);
		out.println("var idx = index;");
		indent(1);
		out.println("if (typeof index.num !== 'undefined') idx = index.num;");
		indent(1);
		out.println("return this.list[idx];\n}");

		out.println("$_List_$.prototype.setValue = function(index, value) {");
		indent(1);
		out.println("var idx = index;");
		indent(1);
		out.println("if (typeof index.num !== 'undefined') idx = index.num;");
		indent(1);
		out.println("this.list[idx] = value;\n}");

		out.println("$_List_$.prototype.length = function() {");
		indent(1);
		out.println("return new $_Integer_$(this.list.length);\n}");

		out.println("$_List_$.prototype.append = function(other) {");
		indent(1);
		out.println("var result = [];");
		indent(1);
		out.println("var cnt = 0;");
		indent(1);
		out.println("for (var i = 0; i < this.list.length; i++)");
		indent(2);
		out.println("result[cnt++] = this.list[i];");
		indent(1);
		out.println("for (var i = 0; i < other.list.length; i++)");
		indent(2);
		out.println("result[cnt++] = other.list[i];");
		indent(1);
		out.println("return new $_List_$(result);\n}");

		out.println("$_List_$.prototype.cast = function(name, fieldList) {");
		out.println("var tmp = [];");
		out.println("for (var i = 0; i < this.list.length; i++)");
		out.println("tmp[i] = this.list[i].cast(name, fieldList);");
		out.println("return new $_List_$(tmp)");
		out.println("}");

		out.println("$_List_$.prototype.clone = function() {");
		indent(1);
		out.println("var clist = [];");
		indent(1);
		out.println("for (var i = 0; i < this.list.length; i++) {");
		indent(2);
		out.println("var elem = this.list[i];");
		indent(2);
		out.println("if (elem instanceof $_List_$ || elem instanceof $_Record_$) {");
		indent(3);
		out.println("elem = elem.clone();\n}");
		indent(2);
		out.println("clist[i] = elem;");
		indent(1);
		out.println("}");
		indent(1);
		out.println("return new $_List_$(clist);\n}");

		out.println("$_List_$.prototype.toString = function() {");
		indent(1);
		out.println("var str = '[';");
		indent(1);
		out.println("var first = true;");
		indent(1);
		out.println("for (var i = 0; i < this.list.length; i++) {");
		indent(2);
		out.println("if (!first)");
		indent(3);
		out.println("str += ', ';");
		indent(2);
		out.println("first = false;");
		indent(2);
		out.println("var val = this.list[i];");
		indent(2);
		out.println("str += val;");
		indent(1);
		out.println("}");
		indent(1);
		out.println("str += ']';");
		indent(1);
		out.println("return str;\n}");

		out.println("$_List_$.prototype.equals = function(other) {");;
		out.println("if (!(other instanceof $_List_$)) return false;");
		out.println("if (this.length().num !== other.length().num) return false;");

		out.println("for (var i = 0; i < this.list.length; i++) {");
		out.println("if (!($_equals_$(this.list[i], other.list[i], true)))");
		out.println("return false;\n}");
		out.println("return true;\n}\n");

	}

	private void setUpBinaryFunctions() {

		//First, the range function
		out.println("function $_range_$(lower, upper) {");
		indent(1);
		out.println("var low = lower;");
		indent(1);
		out.println("var up = upper;");
		indent(1);
		out.println("if (typeof low.num !== 'undefined') low = lower.num;");
		indent(1);
		out.println("if (typeof up.num !== 'undefined') up = upper.num;");
		indent(1);
		out.println("var $_result_$ = [];");
		indent(1);
		out.println("var $_count_$ = 0;");
		indent(1);
		out.println("for (var $_tmp_$ = low; $_tmp_$ < up; $_tmp_$++) {");
		indent(2);
		out.println("$_result_$[$_count_$] = new $_Integer_$($_tmp_$);");
		indent(2);
		out.println("$_count_$++;");
		indent(1);
		out.println("}");
		indent(1);
		out.println("return $_result_$;\n}\n");

		//Next, the append function
		out.println("function $_append_$(left, right) {");
		indent(1);
		out.println("if (left instanceof String || typeof left === 'string') {");
		indent(2);
		out.println("var other = right;");
		indent(2);
		out.println("return left.concat(other.toString());");
		indent(1);
		out.println("}");
		indent(1);
		out.println("else return left.append(right);\n}\n");

		//Finally, the 3 comparison functions - =/!=, >/>= and </<=
		out.println("function $_equals_$(lhs, rhs, isEqual) {");
		indent(1);
		out.println("var left = lhs;");
		indent(1);
		out.println("if (typeof left.num !== 'undefined') left = left.num;");
		indent(1);
		out.println("else if (left instanceof String) left = left.valueOf();");
		indent(1);
		out.println("var right = rhs;");
		indent(1);
		out.println("if (typeof right.num !== 'undefined') right = right.num;");
		indent(1);
		out.println("else if (right instanceof String) right = right.valueOf();");
		out.println("if (left instanceof $_List_$) return left.equals(right);");
		indent(1);
		out.println("if (isEqual) return (left === right);");
		indent(1);
		out.println("else return (left !== right);\n}\n");

		out.println("function $_lt_$(lhs, rhs, isEqual) {");
		indent(1);
		out.println("var left = lhs;");
		indent(1);
		out.println("if (typeof left.num !== 'undefined') left = left.num;");
		indent(1);
		out.println("var right = rhs;");
		indent(1);
		out.println("if (typeof right.num !== 'undefined') right = right.num;");
		indent(1);
		out.println("if (isEqual) return (left <= right);");
		indent(1);
		out.println("else return (left < right);\n}\n");

		out.println("function $_gt_$(lhs, rhs, isEqual) {");
		indent(1);
		out.println("var left = lhs;");
		indent(1);
		out.println("if (typeof left.num !== 'undefined') left = left.num;");
		indent(1);
		out.println("var right = rhs;");
		indent(1);
		out.println("if (typeof right.num !== 'undefined') right = right.num;");
		indent(1);
		out.println("if (isEqual) return (left >= right);");
		indent(1);
		out.println("else return (left > right);\n}\n");
	}

	private void setUpNumberFunctions() {

		// Create a Float function to represent a WyScript Real
		out.println("function $_Float_$(i) {");
		indent(1);
		out.println("if (typeof i.num !== 'undefined') this.num = i.num;");
		indent(1);
		out.println("else this.num = i;");
		indent(1);
		out.println("this.type = 'real';\n}\n");
		out.println("$_Float_$.prototype.add = function(other) {");
		indent(1);
		out.println("return new $_Float_$(this.num + other.num);\n}");
		out.println("$_Float_$.prototype.sub = function(other) {");
		indent(1);
		out.println("return new $_Float_$(this.num - other.num);\n}");
		out.println("$_Float_$.prototype.mul = function(other) {");
		indent(1);
		out.println("return new $_Float_$(this.num * other.num);\n}");
		out.println("$_Float_$.prototype.div = function(other) {");
		indent(1);
		out.println("return new $_Float_$(this.num / other.num);\n}");
		out.println("$_Float_$.prototype.rem = function(other) {");
		indent(1);
		out.println("return new $_Float_$(this.num % other.num);\n}");
		out.println("$_Float_$.prototype.cast = function() {");
		indent(1);
		out.println("return new $_Integer_$(this.num);\n}");
		out.println("$_Float_$.prototype.toString = function() {");
		indent(1);
		out.println("var tmp = this.num.toString();");
		out.println("if (tmp.indexOf('.') === -1)");
		out.println("tmp += '.0';");
		out.println("return tmp;\n}\n");

		// And an Integer function for a WyScript int
		out.println("function $_Integer_$(i) {");
		indent(1);
		out.println("this.type = 'int';");
		indent(1);
		out.println("if (typeof i.num !== 'undefined') this.num = ~~(i.num);");
		indent(1);
		out.println("else this.num = ~~i;\n}\n");
		out.println("$_Integer_$.prototype.add = function(other) {");
		indent(1);
		out.println("if (other instanceof $_Integer_$)");
		indent(2);
		out.println("return new $_Integer_$(this.num + other.num);");
		indent(1);
		out.println("else return new $_Float_$(this.num + other.num);\n}");
		out.println("$_Integer_$.prototype.sub = function(other) {");
		indent(1);
		out.println("if (other instanceof $_Integer_$)");
		indent(2);
		out.println("return new $_Integer_$(this.num - other.num);");
		indent(1);
		out.println("else return new $_Float_$(this.num - other.num);\n}");
		out.println("$_Integer_$.prototype.mul = function(other) {");
		indent(1);
		out.println("if (other instanceof $_Integer_$)");
		indent(2);
		out.println("return new $_Integer_$(this.num * other.num);");
		indent(1);
		out.println("else return new $_Float_$(this.num * other.num);\n}");
		out.println("$_Integer_$.prototype.div = function(other) {");
		indent(1);
		out.println("var tmp = this.num / other.num;");
		indent(1);
		out.println("if (other instanceof $_Integer_$) {");
		indent(2);
		out.println("return new $_Integer_$(~~tmp);\n}");
		indent(1);
		out.println("else return new $_Float_$(this.num / other.num);\n}");
		out.println("$_Integer_$.prototype.rem = function(other) {");
		indent(1);
		out.println("if (other instanceof $_Integer_$)");
		indent(2);
		out.println("return new $_Integer_$(this.num % other.num);");
		indent(1);
		out.println("else return new $_Float_$(this.num % other.num);\n}");
		out.println("$_Integer_$.prototype.cast = function() {");
		indent(1);
		out.println("return new $_Float_$(this.num);\n}");
		out.println("$_Integer_$.prototype.toString = function() {");
		indent(1);
		out.println("return this.num.toFixed()\n}\n");
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
		} else if (stmt instanceof Expr.Invoke) {
			indent(indent);
			write((Expr.Invoke)stmt);
			out.println(";");
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

		Type t = stmt.getLhs().attribute(Attribute.Type.class).type;

		if (t instanceof Type.Char && stmt.getLhs() instanceof Expr.IndexOf) {
			write(((Expr.IndexOf)stmt.getLhs()).getSource());
			out.print(" = $_stringIndexReplace_$(");
			write(((Expr.IndexOf)stmt.getLhs()).getSource());
			out.print(", ");
			write(((Expr.IndexOf)stmt.getLhs()).getIndex());
			out.print(", ");
			write(stmt.getRhs());
			out.print(")");
		}
		else if (stmt.getLhs() instanceof Expr.IndexOf) {
			write(((Expr.IndexOf)stmt.getLhs()).getSource());
			out.print(".setValue(");
			write(((Expr.IndexOf)stmt.getLhs()).getIndex());
			out.print(", ");
			write(stmt.getRhs());
			out.print(")");
		}
		else if (stmt.getLhs() instanceof Expr.RecordAccess) {
			write(((Expr.RecordAccess)stmt.getLhs()).getSource());
			out.print(".setValue('");
			out.print(((Expr.RecordAccess)stmt.getLhs()).getName());
			out.print("', ");
			write(stmt.getRhs());
			out.print(")");
		}
		else {
			write(stmt.getLhs());
			out.print(" = ");
			write(stmt.getRhs());

			//Handle pass by value
			if (t instanceof Type.List || t instanceof Type.Record)
				out.print(".clone()");
		}
	}


	public void write(Stmt.Print stmt) {
		out.print("sysout.println(");
		write(stmt.getExpr());
		out.print(".toString())");
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
		Type t = init.attribute(Attribute.Type.class).type;
		if(init != null) {
			out.print(" = ");
			write(init);
			if (t instanceof Type.List || t instanceof Type.Record)
				out.print(".clone()");
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
		} else if(expr instanceof Expr.Is) {
			write((Expr.Is)expr);
		}
		else {
			internalFailure("unknown expression encountered (" + expr + ")", file.filename,expr);
		}
	}

	public void write(Expr.Binary expr) {

		//Have to handle the case where
		//Working with numbers - must call the method on the JavaScript object

		//Need to handle the nasty left recursive case for maths operators
		Expr.BOp op = expr.getOp();
		if (expr.getRhs() instanceof Expr.Binary && (
				   op == Expr.BOp.ADD || op == Expr.BOp.SUB
				|| op == Expr.BOp.MUL || op == Expr.BOp.DIV
				|| op == Expr.BOp.REM)) {

			Expr.Binary bin = (Expr.Binary) expr.getRhs();
			Expr.BOp otherOp = bin.getOp();

			switch(otherOp) {

			case ADD:
			case DIV:
			case MUL:
			case REM:
			case SUB:
				Expr.Binary lhsExpr = new Expr.Binary(op, expr.getLhs(), bin.getLhs());
				Expr.Binary newExpr = new Expr.Binary(otherOp, lhsExpr, bin.getRhs());
				write(newExpr);
				return;

			default:
				break;
			}
		}
		switch (expr.getOp()) {

		case APPEND:
			out.print("($_append_$(");
			write(expr.getLhs());
			out.print(", ");
			write(expr.getRhs());
			out.print("))");
			return;

		case RANGE:
			out.print("($_range_$(");
			write(expr.getLhs());
			out.print(", ");
			write(expr.getRhs());
			out.print("))");
			return;

		case AND:
		case OR:
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
		if (t instanceof Type.Record || t instanceof Type.List) {

			write(expr.getSource());
			out.print(".cast(");

			List<String> typeList = getRecordTypeList(expr.getSource().attribute(Attribute.Type.class).type, expr.getType());

			if (typeList.isEmpty()) {
				out.print("'', [])");
			}
			else {
				out.print("'" + typeList.get(0) +"', [");
				boolean first = true;
				for (int i = 1; i < typeList.size(); i++) {
					if(!first)
						out.print(", ");
					first = false;
					out.print("'" + typeList.get(i) + "'");
				}
				out.print("])");
			}
		}
		else if (t instanceof Type.Real || t instanceof Type.Int) {
			write(expr.getSource());
			out.print(".cast()");
		}
		else write(expr.getSource());
	}

	private List<String> getRecordTypeList(Type actual, Type castType) {
		if (actual.equals(castType))
			return new ArrayList<String>();

		Type current = castType;
		while (current instanceof Type.Named)
			current = userTypes.get(((Type.Named)current).getName());

		Type temp = actual;
		while (temp instanceof Type.Named)
			temp = userTypes.get(((Type.Named)temp).getName());

		String castedName = "";
		Type casted = null;
		List<String> fieldList = new ArrayList<String>();

		outer: while (casted == null) {
			if (temp instanceof Type.List) {
				temp = ((Type.List)temp).getElement();
				current = ((Type.List)current).getElement();
			}
			else if (temp instanceof Type.Record) {
				Type.Record actualRecord = (Type.Record)temp;
				Type.Record currentRecord = (Type.Record)current;
				for (String s : actualRecord.getFields().keySet()) {
					if (actualRecord.getFields().get(s).equals(currentRecord.getFields().get(s)))
						continue;
					else {
						current = currentRecord.getFields().get(s);
						temp = actualRecord.getFields().get(s);
						if (current instanceof Type.Record || current instanceof Type.List) {
							fieldList.add(s);
							continue outer;
						}
						else {
							casted = temp;
							castedName = s;
							break outer;
						}
					}
				}
				break; //Signals a 'pointless' cast
			}
			else return new ArrayList<String>(); //Should be dead code

		}
		if (casted == null)
			return new ArrayList<String>(); //pointless cast, no difference in types
		else {
			fieldList.add(0, castedName);
			return fieldList;
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
		}
		else if (val instanceof Character) {
			out.print("'");
			out.print(val);
			out.print("'");
		}
		else
			out.print(val);
	}

	public void write(Expr.IndexOf expr) {
		out.print("($_indexOf_$(");
		write(expr.getSource());
		out.print(",");
		write(expr.getIndex());
		out.print("))");
	}

	public void write(Expr.Invoke expr) {
		out.print(expr.getName() + "(");
		boolean firstTime=true;
		for(Expr arg : expr.getArguments()) {
			Type t = arg.attribute(Attribute.Type.class).type;
			if(!firstTime) {
				out.print(",");
			}
			firstTime=false;
			write(arg);
			if (t instanceof Type.List || t instanceof Type.Record)
				out.print(".clone()");
		}
		out.print(")");
	}

	public void write(Expr.ListConstructor expr) {
		out.print("new $_List_$(");
		out.print("[");
		boolean firstTime=true;
		for(Expr arg : expr.getArguments()) {
			if(!firstTime) {
				out.print(",");
			}
			firstTime=false;
			write(arg);
		}
		out.print("])");
	}

	public void write(Expr.RecordAccess expr) {
		write(expr.getSource());
		out.print(".getValue('" + expr.getName() + "')");
	}

	public void write(Expr.RecordConstructor expr) {
		out.print("new $_Record_$(");
		out.print("[");
		boolean firstTime=true;
		for(Pair<String,Expr> p : expr.getFields()) {
			if(!firstTime) {
				out.print(",");
			}
			firstTime=false;
			out.print("'" + p.first() + "'");
		}
		out.print("], [");
		firstTime=true;
		for(Pair<String,Expr> p : expr.getFields()) {
			if(!firstTime) {
				out.print(",");
			}
			firstTime=false;
			write(p.second());
		}
		out.print("])");
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
			out.print(".num))");
			break;
		case LENGTHOF:
			out.print("$_length_$(");
			write(expr.getExpr());
			out.print(")");
		}
		out.print(")");
	}

	public void write(Expr.Variable expr) {
		out.print(expr.getName());
	}

	public void write(Expr.Is expr) {
		//TODO: Sort out this class malarkey
	}

	public void indent(int indent) {
		for(int i=0;i!=indent;++i) {
			out.print("    ");
		}
	}
}
