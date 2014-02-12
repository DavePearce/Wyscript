package wyscript.io;

import static wyscript.util.SyntaxError.internalFailure;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;


import wyscript.lang.*;
import wyscript.util.*;

/**
 * An extended interpreter - instead of outputting directly to the
 * console, converts a given WyScriptFile into an equivalent JavaScript
 * file.
 *
 */
public class JavaScriptFileWriter {
	private PrintStream out;
	private WyscriptFile file;
	private HashMap<String, Type> userTypes;

	private int forCount = 0;
	private int switchCount = 0;	//Used to prevent issues with temporary variables

	public JavaScriptFileWriter(File file) throws IOException {
		this.out = new PrintStream(new FileOutputStream(file));
	}

	public void close() {
		out.close();
	}

	/**
	 * Writes a Wyscript File into a Javascript file
	 */
	public void write(WyscriptFile wf) {
		this.file = wf;
		userTypes = new HashMap<String, Type>();

		//Next, sort out constants and named types
		for (WyscriptFile.Decl declaration : wf.declarations) {
			if (declaration instanceof WyscriptFile.ConstDecl) {
				write((WyscriptFile.ConstDecl)declaration);
			}

			//Don't write type declarations, just store them so we can find the type later
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
	 * Writes a constant declaration -
	 * this declares a global variable
	 */
	public void write(WyscriptFile.ConstDecl cd) {
		Type t = cd.constant.attribute(Attribute.Type.class).type;
		out.print("var "+ cd.name() + " = ");

		if (t instanceof Type.Real) {
			out.print("new Wyscript.Float(");
			write(cd.constant);
			out.print(")");
		}
		else if (t instanceof Type.Int) {
			out.print("new Wyscript.Integer(");
			write(cd.constant);
			out.print(")");
		}
		else
			write(cd.constant);
		out.println(";");
	}

	/**
	 * Writes a function - this writes an equivalent
	 * javascript function
	 */
	public void write(WyscriptFile.FunDecl fd) {

		//Don't need to write native functions, they will be (or should be) already declared
		if (fd.Native)
			return;

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
		write(fd.statements, 1, null);
		out.println("}");
	}

	public void write(List<Stmt> statements, int indent, Expr expr) {
		for(Stmt s : statements) {
			write(s, indent, expr);
		}
	}

	public void write(Stmt stmt, int indent, Expr expr) {

		if(stmt instanceof Stmt.Atom) {
			indent(indent);
			writeAtom((Stmt.Atom) stmt);
			out.println(";");
		} else if(stmt instanceof Stmt.Assign) {
			write((Stmt.Assign)stmt, indent);
		} else if(stmt instanceof Stmt.IfElse) {
			write((Stmt.IfElse) stmt, indent, expr);
		} else if(stmt instanceof Stmt.OldFor) {
			write((Stmt.OldFor) stmt, indent, expr);
		} else if(stmt instanceof Stmt.While) {
			write((Stmt.While) stmt, indent, expr);
		} else if (stmt instanceof Stmt.For) {
			write((Stmt.For) stmt, indent, expr);
		} else if (stmt instanceof Expr.Invoke) {
			indent(indent);
			write((Expr.Invoke)stmt);
			out.println(";");
		} else if (stmt instanceof Stmt.Switch) {
			write((Stmt.Switch)stmt, indent);
		}

		//Handle the next statement, which just sets the control variable
		//and then causes the switch's enclosing loop to repeat
		else if (stmt instanceof Stmt.Next) {
			indent(indent);
			out.print("$WySwitch" + (switchCount-1) + " = ");

			if (expr == null)
				out.print("'$DEFAULT'");

			else
				write(expr);

			out.println(";");
			indent(indent);
			out.println("continue $label" + (switchCount-1) + ";");
		}
		else {
			internalFailure("unknown statement encountered (" + stmt + ")", file.filename,stmt);
		}
	}

	public void write(Stmt.IfElse stmt, int indent, Expr expr) {
		indent(indent);
		out.print("if(");
		write(stmt.getCondition());
		out.println(") {");
		write(stmt.getTrueBranch(),indent+1, expr);
		indent(indent);
		out.println("}");

		for (Expr e : stmt.getAltExpressions()) {
			indent(indent);
			out.print("else if(");
			write(e);
			out.println(") {");
			write(stmt.getAltBranch(e), indent+1, expr);
			indent(indent);
			out.println("}");
		}

		if(stmt.getFalseBranch().size() > 0) {
			indent(indent);
			out.println("else {");
			write(stmt.getFalseBranch(),indent+1, expr);
			indent(indent);
			out.println("}");
		}

	}

	public void write(Stmt.OldFor stmt, int indent, Expr expr) {
		indent(indent);
		out.print("for(");
		writeAtom(stmt.getDeclaration());
		out.print(";");
		write(stmt.getCondition());
		out.print(";");
		writeAtom(stmt.getIncrement());
		out.println(") {");
		write(stmt.getBody(),indent+1, expr);
		indent(indent);
		out.println("}");
	}

	/**
	 * Writes a for each loop. This is converted to a classical for loop,
	 * with extra variables stored inside an object inside the helper
	 * Wyscript.funcs object.
	 *
	 */
	public void write(Stmt.For stmt, int indent, Expr expr) {

		//Use the value of forCount to determine the name of the temp variables
		String name = "$WyTmp" + forCount++;
		indent (indent);
		out.println("var " + name + " = {}");
		indent(indent);
		out.print(name + ".list = ");
		write(stmt.getSource());
		if ((stmt.getSource() instanceof Expr.Binary) && ((Expr.Binary)stmt.getSource()).getOp() == Expr.BOp.RANGE);
		else {
			//Must be accessing a Wyscript.List, by either a constructor or variable
			//so we want to access the array in that Wyscript.List
			out.print(".list");
		}
		out.println(";");
		indent(indent);
		out.println(name + ".count = 0;");
		indent(indent);

		//Simulate a for-each loop by iterating over the list, and defining the index value to be equal
		//to the element at the current index
		out.print("for(" + name + ".count = 0; ");
		out.print(name + ".count < " + name + ".list.length().num; ");
		out.println(name + ".count++) {");
		indent(indent+1);
		out.println("var " + stmt.getIndex().getName() + " = " + name + ".list.getValue(" + name + ".count);");
		write(stmt.getBody(),indent+1, expr);
		indent(indent);
		out.println("}");
	}

	public void write(Stmt.While stmt, int indent, Expr expr) {
		indent(indent);
		out.print("while(");
		write(stmt.getCondition());
		out.println(") {");
		write(stmt.getBody(),indent+1, expr);
		indent(indent);
		out.println("}");
	}

	public void write(Stmt.Switch stmt, int indent) {
		indent(indent);
		//Need to make a labeled loop surrounding switch to simulate explicit fallthrough
		out.print("var $WySwitch" + switchCount + " = ");
		write(stmt.getExpr());
		out.println(";");
		indent(indent);
		out.println("$label" + switchCount++ + ": while(true) {");

		//Now write the actual switch body
		writeSwitchStatements(stmt.cases(), indent+1);
		indent(indent);
		out.println("}");

		//Reset the nested switch count, and delete the property
		switchCount--;
	}

	/**
	 * Writes a WyScript switch statement as a series of else if statements, with
	 * the default providing the role of the optional else block.
	 */
	private void writeSwitchStatements(List<Stmt.SwitchStmt> block, int indent) {
		boolean first = true;
		boolean hasDef = false;
		int defIndex = -1;

		for (int i = 0; i < block.size(); i++) {

			Stmt.SwitchStmt stmt = block.get(i);

			if (stmt instanceof Stmt.Case) {
				indent(indent);

				if (!first) {
					out.print("else ");
				}

				first = false;

				Stmt.Case c = (Stmt.Case)stmt;

				out.print("if(Wyscript.equals($WySwitch" + (switchCount-1) + ", ");
				write(c.getConstant());
				out.println(", true)) {");
			}
			else {
				hasDef = true;
				defIndex = i;
				continue;
			}

			//Need to find what the next element is for fallthrough
			Expr expr = null;
			if ( i < block.size() -1) {
				Stmt.SwitchStmt next = block.get(i+1);
				if (next instanceof Stmt.Case)
					expr = ((Stmt.Case)next).getConstant();
				else expr = null;
			}

			if (stmt instanceof Stmt.Case)
				write(((Stmt.Case)stmt).getStmts(), indent+1, expr);

			//Finally, break the switch - if a next was used it will be evaluated before this is reached
			indent(indent+1);
			out.println("break $label" + (switchCount-1) + ";");
			indent(indent);
			out.println("}\n");
		}

		//Add a default statement that breaks if one doesn't exist
		indent(indent);

		//Need to check for the case where the switch is empty/only contains a default - in that case
		//we omit the surrounding else block
		if (block.size() > 1)
			out.println("else {");
		if (hasDef) {
			Expr defExpr = null;
			if (defIndex < block.size() -1)
				defExpr = (((Stmt.Case)block.get(defIndex+1)).getConstant());
			write(((Stmt.Default)block.get(defIndex)).getStmts(), indent+1, defExpr);
		}
		indent(indent+1);
		out.println("break $label" + (switchCount-1) + ";");

		if (block.size() > 1) {
			indent(indent);
			out.println("}\n");
		}

	}

	public void writeAtom(Stmt stmt) {
		if(stmt instanceof Stmt.Print) {
			write((Stmt.Print) stmt);
		} else if(stmt instanceof Stmt.Return) {
			write((Stmt.Return) stmt);
		} else if(stmt instanceof Stmt.VariableDeclaration) {
			write((Stmt.VariableDeclaration) stmt);
		} else {
			internalFailure("unknown statement encountered (" + stmt + ")", file.filename,stmt);
		}
	}

	public void write(Stmt.Assign stmt, int indent) {
		Type t = stmt.getLhs().attribute(Attribute.Type.class).type;

		//Special case for mutating a string, which is illegal in javascript
		if (t instanceof Type.Char && stmt.getLhs() instanceof Expr.IndexOf) {
			indent(indent);
			write(((Expr.IndexOf)stmt.getLhs()).getSource());
			out.print(" = ");
			write(((Expr.IndexOf)stmt.getLhs()).getSource());
			out.print(".assign(");
			write(((Expr.IndexOf)stmt.getLhs()).getIndex());
			out.print(", ");
			write(stmt.getRhs());
			out.println(");");
		}
		//Must use the library function to mutate a list
		else if (stmt.getLhs() instanceof Expr.IndexOf) {
			indent(indent);
			write(((Expr.IndexOf)stmt.getLhs()).getSource());
			out.print(".setValue(");
			write(((Expr.IndexOf)stmt.getLhs()).getIndex());
			out.print(", ");
			write(stmt.getRhs());
			out.println(");");
		}
		//Must use a library function to mutate a record
		else if (stmt.getLhs() instanceof Expr.RecordAccess) {
			indent(indent);
			write(((Expr.RecordAccess)stmt.getLhs()).getSource());
			out.print(".setValue('");
			out.print(((Expr.RecordAccess)stmt.getLhs()).getName());
			out.print("', ");
			write(stmt.getRhs());
			out.println(");");
		}
		//Must use a library function for a dereference assignment
		else if (stmt.getLhs() instanceof Expr.Deref) {
			indent(indent);
			write(((Expr.Deref)stmt.getLhs()).getExpr());
			out.print(".setValue(");
			write(stmt.getRhs());
			out.println(");");
		}
		//Must use a library function for a tuple assignment
		else if (stmt.getLhs() instanceof Expr.Tuple) {
			Expr.Tuple tuple = (Expr.Tuple) stmt.getLhs();

			indent(indent);
			out.print("var $WyscriptTupleVal = ");
			write(stmt.getRhs());
			out.println(";");

			for (int i = 0; i < tuple.getExprs().size(); i++) {
				indent(indent);
				write(tuple.getExprs().get(i));
				out.print(" = $WyscriptTupleVal");
				out.println(".values[" + i + "];");
			}
		}
		else {
			indent(indent);
			write(stmt.getLhs());
			out.print(" = ");
			write(stmt.getRhs());

			//Handle pass by value
			if (t instanceof Type.List || t instanceof Type.Record || t instanceof Type.Tuple)
				out.println(".clone();");
			else
				out.println(";");
		}
	}


	public void write(Stmt.Print stmt) {
		out.print("Wyscript.print(");
		write(stmt.getExpr());
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
			Type t = init.attribute(Attribute.Type.class).type;
			out.print(" = ");
			write(init);
			if (t instanceof Type.List || t instanceof Type.Record || t instanceof Type.Tuple)
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
		} else if(expr instanceof Expr.Deref) {
			write((Expr.Deref)expr);
		} else if(expr instanceof Expr.New) {
			write((Expr.New)expr);
		} else if(expr instanceof Expr.Tuple) {
			write((Expr.Tuple)expr);
		}

		else {
			internalFailure("unknown expression encountered (" + expr + ")", file.filename,expr);
		}
	}

	public void write(Expr.Binary expr) {

		switch (expr.getOp()) {

		case APPEND:
			write(expr.getLhs());
			out.print(".append(");
			write(expr.getRhs());
			out.print(")");
			return;

		case RANGE:
			out.print("(Wyscript.range(");
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
			out.print("(Wyscript.gt(");
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
			out.print("(Wyscript.lt(");
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
			out.print("(Wyscript.equals(");
			write(expr.getLhs());
			out.print(", ");
			write(expr.getRhs());
			out.print(", false))");
			return;
		case EQ:
			out.print("(Wyscript.equals(");
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

		out.print("Wyscript.cast(");
		write(expr.getType());
		out.print(", ");
		write(expr.getSource());
		out.print(")");
	}

	/**
	 * Removes any references to named types in a given type, and returns the resulting type
	 */
	private Type convertNamedType(Type t) {
		Type current = t;
		while(current instanceof Type.Named)
			current = userTypes.get(current.toString());

		//Clone the list, and replace any named types in the list's element type
		if (current instanceof Type.List) {
			return new Type.List(convertNamedType(((Type.List) current).getElement()));
		}

		//Clone the record and step through its fields, replacing any named types found
		else if (current instanceof Type.Record) {
			HashMap<String, Type> fields = new HashMap<String, Type>(((Type.Record)current).getFields());
			for (String s : fields.keySet()) {
				fields.put(s, convertNamedType(fields.get(s)));
			}
			return new Type.Record(fields);
		}

		//Clone the tuple and step through it, replacing any named types found
		else if (current instanceof Type.Tuple) {
			List<Type> types = new ArrayList<Type>(((Type.Tuple)current).getTypes());
			List<Type> newTypes = new ArrayList<Type>();
			for (Type type : types) {
				newTypes.add(convertNamedType(type));
			}
			return new Type.Tuple(newTypes);
		}

		//Clone the union and step through its bounds, replacing any named types found
		else if (current instanceof Type.Union) {
			List<Type> newBounds = new ArrayList<Type>();
			for (Type type : ((Type.Union)current).getBounds()) {
				newBounds.add(convertNamedType(type));
			}
			return new Type.Union(newBounds);
		}

		else return current;
	}

	public void write(Expr.Constant expr) {

		Object val = expr.getValue();

		if (val instanceof Double) {
			out.print("new Wyscript.Float(");
			out.print(val + ")");
		}
		else if (val instanceof Integer) {
			out.print("new Wyscript.Integer(");
			out.print(val + ")");
		}
		else if (val instanceof StringBuffer) {
			String s = ((StringBuffer) val).toString();
			out.print("new Wyscript.String('");
			out.print(s);
			out.print("')");
		}
		else if (val instanceof Character) {
			out.print("new Wyscript.Char('");
			out.print(val);
			out.print("')");
		}
		else
			out.print(val);
	}

	public void write(Expr.IndexOf expr) {
		write(expr.getSource());
		out.print(".getValue(");
		write(expr.getIndex());
		out.print(")");
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
			if (t instanceof Type.List || t instanceof Type.Record || t instanceof Type.Tuple)
				out.print(".clone()");
		}
		out.print(")");
	}

	public void write(Expr.ListConstructor expr) {

		Type t = expr.attribute(Attribute.Type.class).type;
		t = convertNamedType(t);

		//Create a list object, and pass it its type
		out.print("new Wyscript.List(");
		out.print("[");
		boolean firstTime=true;
		for(Expr arg : expr.getArguments()) {
			if(!firstTime) {
				out.print(",");
			}
			firstTime=false;
			write(arg);
		}
		out.print("], ");
		write(t);
		out.print(")");
	}

	public void write(Expr.RecordAccess expr) {
		write(expr.getSource());
		out.print(".getValue('" + expr.getName() + "')");
	}

	public void write(Expr.RecordConstructor expr) {

		Type t = expr.attribute(Attribute.Type.class).type;
		t = convertNamedType(t);

		//Create a record object, passing it two ordered arrays, the first of the names of its fields,
		//the second of the values of those fields
		out.print("new Wyscript.Record(");
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
		out.print("], ");
		write(t);
		out.print(")");
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
				out.print("new Wyscript.Integer(");
			else
				out.print("new Wyscript.Float(");
			out.print(expr.getOp() + "(");
			write(expr.getExpr());
			out.print(".num))");
			break;
		case LENGTHOF:
			write(expr.getExpr());
			out.print(".length()");
		}
		out.print(")");
	}

	public void write(Expr.Variable expr) {
		out.print(expr.getName());
	}

	public void write(Expr.Is expr) {

		Type t = expr.getRhs();
		t = convertNamedType(t);

		out.print("Wyscript.is(");
		write(expr.getLhs());
		out.print(", ");
		write(t);
		out.print(")");
	}

	public void write(Expr.New expr) {
		out.print("new Wyscript.Ref(");
		write(expr.getExpr());
		out.print(")");
	}

	public void write(Expr.Deref expr) {
		write(expr.getExpr());
		out.print(".deref()");
	}

	public void write(Expr.Tuple expr) {
		out.print("new Wyscript.Tuple([");
		boolean first = true;
		for (Expr e : expr.getExprs()) {
			if (!first)
				out.print(", ");
			first = false;
			write(e);
		}
		out.print("], ");
		write(expr.attribute(Attribute.Type.class).type);
		out.print(")");
	}

	/**
	 * Writes a type - necessary for Record and List objects
	 */
	public void write(Type t) {
		t = convertNamedType(t);
		if (t instanceof Type.Null)
			write((Type.Null)t);
		else if (t instanceof Type.Void)
			write((Type.Void)t);
		else if (t instanceof Type.Bool)
			write((Type.Bool)t);
		else if (t instanceof Type.Int)
			write((Type.Int)t);
		else if (t instanceof Type.Real)
			write((Type.Real)t);
		else if (t instanceof Type.Char)
			write((Type.Char)t);
		else if (t instanceof Type.Strung)
			write((Type.Strung)t);
		else if (t instanceof Type.List)
			write((Type.List)t);
		else if (t instanceof Type.Record)
			write((Type.Record)t);
		else if (t instanceof Type.Union)
			write((Type.Union)t);
		else if (t instanceof Type.Reference)
			write((Type.Reference)t);
		else if (t instanceof Type.Tuple)
			write((Type.Tuple)t);
		else internalFailure("Unknown type encountered: " + t, file.filename, t);
	}

	public void write(Type.Null t) {
		out.print("new Wyscript.Type.Null()");
	}

	public void write(Type.Void t) {
		out.print("new Wyscript.Type.Void()");
	}

	public void write(Type.Bool t) {
		out.print("new Wyscript.Type.Bool()");
	}

	public void write(Type.Int t) {
		out.print("new Wyscript.Type.Int()");
	}

	public void write(Type.Real t) {
		out.print("new Wyscript.Type.Real()");
	}

	public void write(Type.Char t) {
		out.print("new Wyscript.Type.Char()");
	}

	public void write(Type.Strung t) {
		out.print("new Wyscript.Type.String()");
	}

	public void write(Type.List t) {
		out.print("new Wyscript.Type.List(");
		write(t.getElement());
		out.print(")");
	}

	public void write(Type.Record t) {
		out.print("new Wyscript.Type.Record([");
		List<String> names = new ArrayList<String>(t.getFields().keySet());
		Collections.sort(names);

		boolean first = true;
		for (String s : names) {
			if (!first)
				out.print(", ");

			first = false;
			out.print("'" + s + "'");
		}
		out.print("], [");
		first = true;
		for (String s : names) {
			if (!first)
				out.print(", ");

			first = false;
			write(t.getFields().get(s));
		}
		out.print("])");
	}

	public void write(Type.Union t) {
		out.print("new Wyscript.Type.Union([");
		boolean first = true;

		for (Type b : t.getBounds()) {
			if (!first)
				out.print(", ");

			first = false;
			write(b);
		}
		out.print("])");
	}

	public void write(Type.Reference t) {
		out.print("new Wyscript.Type.Reference(");
		write(t.getType());
		out.print(")");
	}

	public void write(Type.Tuple t) {
		out.print("new Wyscript.Type.Tuple([");
		boolean first = true;
		for (Type type : t.getTypes()) {
			if (!first)
				out.print(", ");
			first = false;
			write(type);
		}
		out.print("])");
	}

	public void indent(int indent) {
		for(int i=0;i!=indent;++i) {
			out.print("    ");
		}
	}
}
