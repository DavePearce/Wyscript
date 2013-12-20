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
 * file.
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

	/**
	 * Writes a function - this writes an equivalent
	 * javascript function
	 */
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
		} else if (stmt instanceof Stmt.Switch) {
			write((Stmt.Switch)stmt, indent);
		}
		//Handle the next statement, which just sets the control variable
		//and then causes the switch's enclosing loop to repeat
		else if (stmt instanceof Stmt.Next) {
			indent(indent);
			out.print("$_label_$ = ");

			if (expr == null)
				out.print("$_default_$");

			else
				write(expr);

			out.println(";");
			indent(indent);
			out.println("continue $_loop_$;");
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
		write(stmt.getTrueBranch(),indent+1, null);
		if(stmt.getFalseBranch().size() > 0) {
			indent(indent);
			out.println("} else {");
			write(stmt.getFalseBranch(),indent+1, null);
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
		write(stmt.getBody(),indent+1, null);
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

		//Simulate a for-each loop by iterating over the list, and defining the index value to be equal
		//to the element at the current index
		out.print("for(var $_tmp_$ = 0; $_tmp_$ < $$__tmp__$$.length; $_tmp_$++) {\n");
		indent(indent+1);
		out.println("var " + stmt.getIndex().getName() + " = $$__tmp__$$[$_tmp_$];");
		write(stmt.getBody(),indent+1, null);
		indent(indent);
		out.println("}");
	}

	public void write(Stmt.While stmt, int indent) {
		indent(indent);
		out.print("while(");
		write(stmt.getCondition());
		out.println(") {");
		write(stmt.getBody(),indent+1, null);
		indent(indent);
		out.println("}");
	}

	public void write(Stmt.Switch stmt, int indent) {
		indent(indent);
		//Need to make a labeled loop surrounding switch to simulate explicit fallthrough
		out.print("var $_label_$ = ");
		write(stmt.getExpr());
		out.println(";");
		indent(indent);
		out.println("$_loop_$: while(true) {");

		//Now write the actual switch body
		writeSwitchStatements(stmt.cases(), indent+1);
		indent(indent);
		out.println("}");
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

				out.print("if($_equals_$($_label_$, ");
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
			out.println("break $_loop_$;");
			indent(indent);
			out.println("}\n");
		}

		//Add a default statement that breaks if one doesn't exist
		indent(indent);
		out.println("else {");
		if (hasDef) {
			Expr defExpr = null;
			if (defIndex < block.size() -1)
				defExpr = (((Stmt.Case)block.get(defIndex+1)).getConstant());
			write(((Stmt.Default)block.get(defIndex)).getStmts(), indent+1, defExpr);
		}
		indent(indent+1);
		out.println("break $_loop_$;");
		indent(indent);
		out.println("}\n");

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

		//Special case for mutating a string, which is illegal in javascript
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
		//Must use the library function to mutate a list
		else if (stmt.getLhs() instanceof Expr.IndexOf) {
			write(((Expr.IndexOf)stmt.getLhs()).getSource());
			out.print(".setValue(");
			write(((Expr.IndexOf)stmt.getLhs()).getIndex());
			out.print(", ");
			write(stmt.getRhs());
			out.print(")");
		}
		//Must use a library function to mutate a record
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
		out.print("$_print_$(");
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
		while (t instanceof Type.Named)
			t = userTypes.get(t.toString());

		//Pointless cast
		if (t.equals(expr.getSource().attribute(Attribute.Type.class).type))
			write(expr.getSource());

		//Can call the cast method of a record or list
		if (t instanceof Type.Record || t instanceof Type.List) {

			write(expr.getSource());
			out.print(".cast(");

			List<String> typeList = getRecordTypeList(expr.getSource().attribute(Attribute.Type.class).type, expr.getType());

			if (typeList.isEmpty()) {
				out.print("'', [],'" + t.toString() + "')");
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
				out.print("],'" + t.toString() + "')");
			}
		}
		else if (t instanceof Type.Real || t instanceof Type.Int) {
			write(expr.getSource());
			out.print(".cast('" + t.toString() + "')");
		}
		//This cast has no impact on the program (and has already been type checked)
		else write(expr.getSource());
	}

	/**
	 * Gets the list of fields changed in a cast affecting a record type - the first element
	 * will be the name of the field changed, and the subsequent elements form a list of the fields
	 * that must be traversed to reach the changed field.
	 */
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

		Type t = expr.attribute(Attribute.Type.class).type;
		while (t instanceof Type.Named)
			t = userTypes.get(t.toString());

		//Create a list object, and pass it its type
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
		out.print("],'" + t.toString() +"')");
	}

	public void write(Expr.RecordAccess expr) {
		write(expr.getSource());
		out.print(".getValue('" + expr.getName() + "')");
	}

	public void write(Expr.RecordConstructor expr) {

		Type t = expr.attribute(Attribute.Type.class).type;
		while (t instanceof Type.Named)
			t = userTypes.get(t.toString());

		//Create a record object, passing it two ordered arrays, the first of the names of its fields,
		//the second of the values of those fields
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
		out.print("],'" + t.toString() +"')");
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

		Type t = expr.getRhs();
		while (t instanceof Type.Named)
			t = userTypes.get(t.toString());

		out.print("$_is_$(");
		write(expr.getLhs());
		out.print(",'" + t + "')");
	}

	public void indent(int indent) {
		for(int i=0;i!=indent;++i) {
			out.print("    ");
		}
	}
}
