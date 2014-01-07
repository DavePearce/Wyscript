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

import java.util.*;

import wyscript.lang.*;
import wyscript.lang.Expr.Binary;
import wyscript.par.KernelRunner;
import wyscript.util.Pair;
import wyscript.util.SyntacticElement;
import static wyscript.util.SyntaxError.*;

/**
 * A simple interpreter for WyScript programs, which executes them in their
 * Abstract Syntax Tree form directly. The interpreter is not designed to be
 * efficient in anyway, however it's purpose is to provide a reference
 * implementation for the language.
 *
 * @author David J. Pearce
 *
 */
public class Interpreter {
	private HashMap<String, WyscriptFile.Decl> declarations;
	private WyscriptFile file;
	private HashMap<String, Object> constants;
	private HashMap<String, Type> userTypes;


	public void run(WyscriptFile wf) {
		// First, initialise the map of declaration names to their bodies.
		//Also, initialise any constant values declared in the file
		declarations = new HashMap<String,WyscriptFile.Decl>();
		constants = new HashMap<String, Object>();
		userTypes = new HashMap<String, Type>();

		for(WyscriptFile.Decl decl : wf.declarations) {
			declarations.put(decl.name(), decl);

			if (decl instanceof WyscriptFile.ConstDecl) {
				WyscriptFile.ConstDecl constant = (WyscriptFile.ConstDecl) decl;
				constants.put(constant.name, execute(constant.constant, constants));
			}

			else if (decl instanceof WyscriptFile.TypeDecl) {
				WyscriptFile.TypeDecl type = (WyscriptFile.TypeDecl) decl;
				userTypes.put(type.name(), type.type);
			}
		}
		this.file = wf;

		// Second, pick the main method (if one exits) and execute it
		WyscriptFile.Decl main = declarations.get("main");
		if(main instanceof WyscriptFile.FunDecl) {
			WyscriptFile.FunDecl fd = (WyscriptFile.FunDecl) main;
			execute(fd);
		} else {
			System.out.println("Cannot find a main() function");
		}
	}

	/**
	 * Execute a given function with the given argument values. If the number of
	 * arguments is incorrect, then an exception is thrown.
	 *
	 * @param function
	 *            Function declaration to execute.
	 * @param arguments
	 *            Array of argument values.
	 */
	private Object execute(WyscriptFile.FunDecl function, Object... arguments) {

		// First, sanity check the number of arguments
		if(function.parameters.size() != arguments.length){
			throw new RuntimeException(
					"invalid number of arguments supplied to execution of function \""
							+ function.name + "\"");
		}

		// Second, construct the stack frame in which this function will
		// execute.
		HashMap<String,Object> frame = new HashMap<String,Object>();
		for(int i=0;i!=arguments.length;++i) {
			WyscriptFile.Parameter parameter = function.parameters.get(i);
			frame.put(parameter.name,arguments[i]);
		}

		for (String s : constants.keySet()) {
			frame.put(s, constants.get(s));
		}

		// Third, execute the function body!
		return execute(function.statements,frame);
	}

	private Object execute(List<Stmt> block, HashMap<String,Object> frame) {
		for(int i=0;i!=block.size();i=i+1) {
			Object r = execute(block.get(i),frame);
			if(r != null) {
				return r;
			}
		}
		return null;
	}

	/**
	 * Execute a given statement in a given stack frame.
	 *
	 * @param stmt
	 *            Statement to execute.
	 * @param frame
	 *            Stack frame mapping variables to their current value.
	 * @return
	 */
	private Object execute(Stmt stmt, HashMap<String,Object> frame) {
		if(stmt instanceof Stmt.Assign) {
			return execute((Stmt.Assign) stmt,frame);
		} else if(stmt instanceof Stmt.OldFor) {
			return execute((Stmt.OldFor) stmt,frame);
		} else if(stmt instanceof Stmt.For) {
			return execute((Stmt.For) stmt,frame);
		}else if (stmt instanceof Stmt.ParFor) {
			boundCalculate(((Stmt.ParFor) stmt).getCalc(), frame);
			return execute((Stmt.ParFor)stmt,frame);
		}
		else if(stmt instanceof Stmt.While) {
			return execute((Stmt.While) stmt,frame);
		} else if(stmt instanceof Stmt.IfElse) {
			return execute((Stmt.IfElse) stmt,frame);
		} else if(stmt instanceof Stmt.Return) {
			return execute((Stmt.Return) stmt,frame);
		} else if(stmt instanceof Stmt.VariableDeclaration) {
			return execute((Stmt.VariableDeclaration) stmt,frame);
		} else if(stmt instanceof Stmt.Print) {
			return execute((Stmt.Print) stmt,frame);
		} else if(stmt instanceof Expr.Invoke) {
			return execute((Expr.Invoke) stmt,frame);
		} else if(stmt instanceof Stmt.Switch) {
			return execute((Stmt.Switch) stmt, frame);
		} else if(stmt instanceof Stmt.Next) {
			return execute((Stmt.Next)stmt, frame);
		} else {
			internalFailure("unknown statement encountered (" + stmt + ")", file.filename,stmt);
			return null;
		}
	}

	private Object execute(Stmt.Switch stmt, HashMap<String, Object> frame) {
		Object expr = execute(stmt.getExpr(), frame);

		boolean hasEvaluated = false;
		boolean evaluateNext = false;

		Stmt.Default def = null;

		for (Stmt.SwitchStmt s: stmt.cases()) {
			if (s instanceof Stmt.Default)  {
				def = (Stmt.Default) s;
				if (evaluateNext) {
					evaluateNext = false;
					Object tmp = execute(def.getStmts(), frame);

					if (tmp instanceof Type.Null)
						evaluateNext = true;
						else return tmp;
				}
			}
			else {
				Stmt.Case c = (Stmt.Case) s;
				Object o = execute(c.getConstant(), frame);

				//We've found a match amongst the cases, or fall-through has occurred
				if (o.equals(expr) || evaluateNext) {
					hasEvaluated = true;
					evaluateNext = false;
					Object tmp = execute(c.getStmts(), frame);

					//Fall-through
					if (tmp instanceof Type.Null)
						evaluateNext = true;

					else return tmp;
				}
			}
		}
		if (def != null && !hasEvaluated) {
			Object tmp = execute(def.getStmts(), frame);
			if (!(tmp instanceof Type.Null))
				return tmp;
		}
		return null;
	}

	private Object execute(Stmt.Next stmt, HashMap<String, Object> frame) {
		//Tombstone value to signal to the switch to progress to the next case
		return new Type.Null();
	}

	private Object execute(Stmt.Assign stmt, HashMap<String,Object> frame) {
		Expr lhs = stmt.getLhs();
		if(lhs instanceof Expr.Variable) {
			Expr.Variable ev = (Expr.Variable) lhs;
			Object rhs = execute(stmt.getRhs(),frame);
			// We need to perform a deep clone here to ensure the value
			// semantics used in While are preserved.
			frame.put(ev.getName(),deepClone(rhs));
		} else if(lhs instanceof Expr.RecordAccess) {
			Expr.RecordAccess ra = (Expr.RecordAccess) lhs;
			Map<String,Object> src = (Map) execute(ra.getSource(),frame);
			Object rhs = execute(stmt.getRhs(),frame);
			// We need to perform a deep clone here to ensure the value
			// semantics used in While are preserved.
			src.put(ra.getName(), deepClone(rhs));
		} else if(lhs instanceof Expr.IndexOf) {
			Expr.IndexOf io = (Expr.IndexOf) lhs;
			Object src = execute(io.getSource(),frame);
			Integer idx = (Integer) execute(io.getIndex(),frame);
			Object rhs = execute(stmt.getRhs(),frame);
			if(src instanceof ArrayList) {
				ArrayList<Object> list = (ArrayList) src;
				// We need to perform a deep clone here to ensure the value
				// semantics used in While are preserved.
				list.set(idx,deepClone(rhs));
			} else {
				StringBuffer str = (StringBuffer) src;
				str.setCharAt(idx, (Character) rhs);
			}
		} else {
			internalFailure("unknown lval encountered (" + lhs + ")", file.filename,stmt);
		}

		return null;
	}

	private Object execute(Stmt.OldFor stmt, HashMap<String,Object> frame) {
		execute(stmt.getDeclaration(),frame);
		while((Boolean) execute(stmt.getCondition(),frame)) {
			Object ret = execute(stmt.getBody(),frame);
			if(ret != null) {
				return ret;
			}
			execute(stmt.getIncrement(),frame);
		}
		return null;
	}

	private Object execute(Stmt.For stmt, HashMap<String,Object> frame) {
		List src = (List) execute(stmt.getSource(),frame);
		String index = stmt.getIndex().getName();
		for(Object item : src) {
			frame.put(index, item);
			Object ret = execute(stmt.getBody(),frame);
			if(ret != null) {
				return ret;
			}
		}
		return null;
	}
	private Object execute(Stmt.ParFor stmt, HashMap<String,Object> frame) {
		if (stmt.getRunner() == null) { //the runner is not available, must fail
			InternalFailure.internalFailure("Could not execute ParFor. " +
					"KernelRunner was not found", file.filename, stmt);
		}else {
			Object out = stmt.getRunner().run(frame);
			return out;
		}
		return null;

	}
	private Object execute(Stmt.While stmt, HashMap<String,Object> frame) {
		while((Boolean) execute(stmt.getCondition(),frame)) {
			Object ret = execute(stmt.getBody(),frame);
			if(ret != null) {
				return ret;
			}
		}
		return null;
	}

	private Object execute(Stmt.IfElse stmt, HashMap<String,Object> frame) {
		boolean condition = (Boolean) execute(stmt.getCondition(),frame);
		if(condition) {
			return execute(stmt.getTrueBranch(),frame);
		} else {
			return execute(stmt.getFalseBranch(),frame);
		}
	}

	private Object execute(Stmt.Return stmt, HashMap<String,Object> frame) {
		Expr re = stmt.getExpr();
		if(re != null) {
			return execute(re,frame);
		} else {
			return Collections.EMPTY_SET; // used to indicate a function has returned
		}
	}

	private Object execute(Stmt.VariableDeclaration stmt,
			HashMap<String, Object> frame) {
		Expr re = stmt.getExpr();
		Object value;
		if (re != null) {
			value = execute(re, frame);
		} else {
			value = Collections.EMPTY_SET; // used to indicate a variable has
										   // been declared
		}
		// We need to perform a deep clone here to ensure the value
		// semantics used in While are preserved.
		value = deepClone(value);
		if (stmt.getType() instanceof Type.Real && value instanceof Integer) {
			value = (double)((Integer)value).intValue();
		}
		frame.put(stmt.getName(), deepClone(value));
		return null;
	}

	private Object execute(Stmt.Print stmt, HashMap<String,Object> frame) {
		String str = toString(execute(stmt.getExpr(),frame));
		System.out.println(str);
		return null;
	}

	/**
	 * Execute a given expression in a given stack frame.
	 *
	 * @param expr
	 *            Expression to execute.
	 * @param frame
	 *            Stack frame mapping variables to their current value.
	 * @return
	 */
	public Object execute(Expr expr, HashMap<String,Object> frame) {
		if(expr instanceof Expr.Binary) {
			return execute((Expr.Binary) expr,frame);
		} else if(expr instanceof Expr.Is) {
			return execute((Expr.Is) expr,frame);
		} else if(expr instanceof Expr.Cast) {
			return execute((Expr.Cast) expr,frame);
		} else if(expr instanceof Expr.Constant) {
			return execute((Expr.Constant) expr,frame);
		} else if(expr instanceof Expr.Invoke) {
			return execute((Expr.Invoke) expr,frame);
		} else if(expr instanceof Expr.IndexOf) {
			return execute((Expr.IndexOf) expr,frame);
		} else if(expr instanceof Expr.ListConstructor) {
			return execute((Expr.ListConstructor) expr,frame);
		} else if(expr instanceof Expr.RecordAccess) {
			return execute((Expr.RecordAccess) expr,frame);
		} else if(expr instanceof Expr.RecordConstructor) {
			return execute((Expr.RecordConstructor) expr,frame);
		} else if(expr instanceof Expr.Unary) {
			return execute((Expr.Unary) expr,frame);
		} else if(expr instanceof Expr.Variable) {
			return execute((Expr.Variable) expr,frame);
		} else {
			internalFailure("unknown expression encountered (" + expr + ")", file.filename,expr);
			return null;
		}
	}

	private Object execute(Expr.Binary expr, HashMap<String,Object> frame) {
		// First, deal with the short-circuiting operators first
		Object lhs = execute(expr.getLhs(), frame);

		switch (expr.getOp()) {
		case AND:
			return ((Boolean)lhs) && ((Boolean)execute(expr.getRhs(), frame));
		case OR:
			return ((Boolean)lhs) || ((Boolean)execute(expr.getRhs(), frame));
		}

		// Second, deal the rest.
		Object rhs = execute(expr.getRhs(), frame);
		Expr.BOp op = expr.getOp();

		//Need to handle the nasty left recursive case for maths operators
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
				Expr.Binary newExpr = new Expr.Binary(op, expr.getLhs(), bin.getLhs());
				lhs = execute(newExpr, frame);
				rhs = execute(bin.getRhs(), frame);
				op = otherOp;

			default:
				break;
			}
		}

		switch (op) {
		case ADD:
			if(lhs instanceof Integer) {
				return ((Integer)lhs) + ((Integer)rhs);
			} else {
				return ((Double)lhs) + ((Double)rhs);
			}
		case SUB:

			if(lhs instanceof Integer) {
				return ((Integer)lhs) - ((Integer)rhs);
			} else {
				return ((Double)lhs) - ((Double)rhs);
			}

		case MUL:
			if(lhs instanceof Integer) {
				return ((Integer)lhs) * ((Integer)rhs);
			} else {
				return ((Double)lhs) * ((Double)rhs);
			}
		case DIV:
			if(lhs instanceof Integer) {
				return ((Integer)lhs) / ((Integer)rhs);
			} else {
				return ((Double)lhs) / ((Double)rhs);
			}
		case REM:
			if(lhs instanceof Integer) {
				return ((Integer)lhs) % ((Integer)rhs);
			} else {
				return ((Double)lhs) % ((Double)rhs);
			}
		case EQ:
			return lhs.equals(rhs);
		case NEQ:
			return !lhs.equals(rhs);
		case LT:
			if(lhs instanceof Integer) {
				return ((Integer)lhs) < ((Integer)rhs);
			} else {
				return ((Double)lhs) < ((Double)rhs);
			}
		case LTEQ:
			if(lhs instanceof Integer) {
				return ((Integer)lhs) <= ((Integer)rhs);
			} else {
				return ((Double)lhs) <= ((Double)rhs);
			}
		case GT:
			if(lhs instanceof Integer) {
				return ((Integer)lhs) > ((Integer)rhs);
			} else {
				return ((Double)lhs) > ((Double)rhs);
			}
		case GTEQ:
			if(lhs instanceof Integer) {
				return ((Integer)lhs) >= ((Integer)rhs);
			} else {
				return ((Double)lhs) >= ((Double)rhs);
			}
		case APPEND:
			if(lhs instanceof StringBuffer && rhs instanceof StringBuffer) {
				StringBuffer l = (StringBuffer) lhs;
				return new StringBuffer(l).append((StringBuffer)rhs);
			} else if(lhs instanceof StringBuffer) {
				StringBuffer l = (StringBuffer) lhs;
				return new StringBuffer(l).append(toString(rhs));
			} else if(rhs instanceof StringBuffer) {
				return toString(lhs) + ((StringBuffer)rhs);
			} else if(lhs instanceof ArrayList && rhs instanceof ArrayList) {
				ArrayList<Object> l = (ArrayList<Object>) lhs;
				l = (ArrayList) deepClone(l);
				l.addAll((ArrayList<Object>)rhs);
				return l;
			}
		case RANGE: {
			int start = (Integer) lhs;
			int end = (Integer) rhs;
			ArrayList<Integer> result = new ArrayList<Integer>();
			while(start < end) {
				result.add(start);
				start = start + 1;
			}
			return result;
		}
		}

		internalFailure("unknown binary expression encountered (" + expr + ")",
				file.filename, expr);
		return null;
	}

	private Object execute(Expr.Is expr, HashMap<String, Object> frame) {
		Object lhs = execute(expr.getLhs(), frame);
		return instanceOf(lhs,expr.getRhs());
	}

	private Object execute(Expr.Cast expr, HashMap<String, Object> frame) {
		Object rhs = execute(expr.getSource(), frame);

		return doCast(expr.getType(), rhs, expr.getSource());
	}

	/**
	 * Method that passes cast execution to the appropriate method
	 * for the type of the cast
	 */
	private Object doCast(Type t, Object o, SyntacticElement elem) {

		if (t instanceof Type.List)
			return doListCast((Type.List)t, (ArrayList)o, elem);

		else if(t instanceof Type.Record) {
			return doRecordCast((Type.Record)t, (HashMap)o, elem);
		}
		else if(t instanceof Type.Union) {
			//We trust the type checker has done its job
			return o;
		}
		else if(t instanceof Type.Named) {
			return doCast(userTypes.get(t.toString()), o, elem);
		}

		else return doPrimitiveCast(t, o, elem);

	}

	private Object doRecordCast(Type.Record t, HashMap o, SyntacticElement elem) {
		HashMap result = new HashMap();

		for(Object name : o.keySet()) {
			Object casted = null;
			casted = doCast(t.getFields().get(name), o.get(name), elem);

			result.put(name, casted);
		}

		return result;
	}

	/**
	 * Casts a non-list object to the java equivalent of the given type,
	 * and returns the resulting object.
	 */
	private Object doPrimitiveCast(Type t, Object o, SyntacticElement elem) {
		Class c = getJavaClass(t);

		//Need to have explicit conversions for the number types
		//As Double cannot be cast to Integer, and vice versa

		if (c.equals(Double.class)) {
			Double d = 0.0;

			if (o instanceof Integer) {
				d = ((Integer)o).doubleValue();
				return d;
			}

			else if (o instanceof Double) {
				d = (Double)o;
				return d;
			}

			else {
				//Shouldn't happen, indicates a type failure
				internalFailure("Casting error - cannot cast between types", file.filename, elem);
				return null;
			}
		}

		else if(c.equals(Integer.class)) {

				Integer i = 0;

				if (o instanceof Integer) {
					i = (Integer)o;
					return i;
				}

				else if (o instanceof Double) {
					i = ((Double)o).intValue();
					return i;
				}

				else {
					//Shouldn't happen, indicates a type checking failure
					internalFailure("Casting error - cannot cast between types", file.filename, elem);
					return null;
				}
		}

		//In all other cases, type checker should have paved the way for us
		return c.cast(o);
	}

	/**
	 * Casts all elements in a list to the given type of the list class.
	 * Recursively deals with nested lists.
	 *
	 * @param t 	- The type of the cast
	 * @param list	- The list being casted
	 * @return
	 */
	private ArrayList doListCast(Type.List t, ArrayList list, SyntacticElement elem) {

		Class listType = getJavaClass(t.getElement());
		ArrayList newList = new ArrayList();

		for (Object o : list)
			newList.add(doCast(t.getElement(), o, elem));

		return newList;
	}

	private Object execute(Expr.Constant expr, HashMap<String,Object> frame) {
		return expr.getValue();
	}

	private Object execute(Expr.Invoke expr, HashMap<String, Object> frame) {
		List<Expr> arguments = expr.getArguments();
		Object[] values = new Object[arguments.size()];
		for (int i = 0; i != values.length; ++i) {
			// We need to perform a deep clone here to ensure the value
			// semantics used in While are preserved.
			values[i] = deepClone(execute(arguments.get(i), frame));
		}
		WyscriptFile.FunDecl fun = (WyscriptFile.FunDecl) declarations.get(expr
				.getName());
		return execute(fun, values);
	}

	private Object execute(Expr.IndexOf expr, HashMap<String,Object> frame) {
		Object _src = execute(expr.getSource(),frame);
		int idx = (Integer) execute(expr.getIndex(),frame);
		if(_src instanceof StringBuffer) {
			StringBuffer src = (StringBuffer) _src;
			return src.charAt(idx);
		} else {
			ArrayList<Object> src = (ArrayList<Object>) _src;
			return src.get(idx);
		}
	}

	private Object execute(Expr.ListConstructor expr,
			HashMap<String, Object> frame) {
		List<Expr> es = expr.getArguments();
		ArrayList<Object> ls = new ArrayList<Object>();
		for (int i = 0; i != es.size(); ++i) {
			ls.add(execute(es.get(i), frame));
		}
		return ls;
	}

	private Object execute(Expr.RecordAccess expr, HashMap<String, Object> frame) {
		HashMap<String, Object> src = (HashMap) execute(expr.getSource(), frame);
		return src.get(expr.getName());
	}

	private Object execute(Expr.RecordConstructor expr, HashMap<String,Object> frame) {
		List<Pair<String,Expr>> es = expr.getFields();
		HashMap<String,Object> rs = new HashMap<String,Object>();

		for(Pair<String,Expr> e : es) {
			rs.put(e.first(),execute(e.second(),frame));
		}

		return rs;
	}

	private Object execute(Expr.Unary expr, HashMap<String, Object> frame) {
		Object value = execute(expr.getExpr(), frame);
		switch (expr.getOp()) {
		case NOT:
			return !((Boolean) value);
		case NEG:
			if (value instanceof Double) {
				return -((Double) value);
			} else {
				return -((Integer) value);
			}
		case LENGTHOF:
			if(value instanceof StringBuffer) {
				return ((StringBuffer) value).length();
			} else {
				return ((ArrayList) value).size();
			}
		}

		internalFailure("unknown unary expression encountered (" + expr + ")",
				file.filename, expr);
		return null;
	}

	private Object execute(Expr.Variable expr, HashMap<String,Object> frame) {
		return frame.get(expr.getName());
	}
	/**
	 * Calculates width (number of columns) and length (number of rows)
	 * of a row-indexed array and inject these values into the frame.
	 * @param calc
	 * @param frame
	 * @return
	 */
	private void boundCalculate(Stmt.BoundCalc calc , HashMap<String,Object> frame) {
		Expr outer = calc.getOuter();
		Expr inner = calc.getInner();
		if (outer == null) {
			calc.setLowX(-1);
			calc.setHighX(-1);
		}
		else if (outer instanceof Expr.Binary) {
			Expr.Binary binary = (Binary) outer;
			int left = (Integer) execute(binary.getLhs(), frame);
			int right = (Integer) execute(binary.getRhs(), frame);
			calc.setLowX(left);
			calc.setHighX(right);
		}else{
			//expression must be a list
			Object value = execute(outer, frame);
			if (value instanceof List<?>){
				calc.setLowX(0);
				calc.setHighX(((List<?>) value).size());
			}else{
				InternalFailure.internalFailure("Could not interpret loop expression",
						file.filename, outer);
			}
		}
		if (inner == null) {
			calc.setLowY(-1);
			calc.setHighY(-1);
		}
		else if (inner instanceof Expr.Binary) {
			Expr.Binary binary = (Binary) inner;
			int left = (Integer) execute(binary.getLhs(), frame);
			int right = (Integer) execute(binary.getRhs(), frame);
			calc.setLowY(left);
			calc.setHighY(right);
		}else{
			//expression must be a list
			Object value = execute(inner, frame);
			if (value instanceof List<?>){
				calc.setLowY(0);
				calc.setHighY(((List<?>) value).size());
			}else{
				InternalFailure.internalFailure("Could not interpret loop expression",
						file.filename, inner);
			}
		}
	}
	/**
	 * Perform a deep clone of the given object value. This is either a
	 * <code>Boolean</code>, <code>Integer</code>, <code>Double</code>,
	 * <code>Character</code>, <code>String</code>, <code>ArrayList</code> (for
	 * lists) or <code>HaspMap</code> (for records). Only the latter two need to
	 * be cloned, since the others are immutable.
	 *
	 * @param o
	 * @return
	 */
	private Object deepClone(Object o) {
		if (o instanceof ArrayList) {
			ArrayList<Object> l = (ArrayList) o;
			ArrayList<Object> n = new ArrayList<Object>();
			for (int i = 0; i != l.size(); ++i) {
				n.add(deepClone(l.get(i)));
			}
			return n;
		} else if (o instanceof HashMap) {
			HashMap<String, Object> m = (HashMap) o;
			HashMap<String, Object> n = new HashMap<String, Object>();
			for (String field : m.keySet()) {
				n.put(field, deepClone(m.get(field)));
			}
			return n;
		} else if (o instanceof StringBuffer) {
			StringBuffer sb = (StringBuffer) o;
			return new StringBuffer(sb);
		} else {
			// other cases can be ignored
			return o;
		}
	}

	/**
	 * Convert the given object value to a string. This is either a
	 * <code>Boolean</code>, <code>Integer</code>, <code>Double</code>,
	 * <code>Character</code>, <code>String</code>, <code>ArrayList</code> (for
	 * lists) or <code>HashMap</code> (for records). The latter two must be
	 * treated recursively.
	 *
	 * @param o
	 * @return
	 */
	private String toString(Object o) {
		if (o instanceof ArrayList) {
			ArrayList<Object> l = (ArrayList) o;
			String r = "[";
			for (int i = 0; i != l.size(); ++i) {
				if(i != 0) {
					r = r + ", ";
				}
				r += toString(l.get(i));
			}
			return r + "]";
		} else if (o instanceof HashMap) {
			HashMap<String, Object> m = (HashMap) o;
			String r = "{";
			boolean firstTime = true;
			ArrayList<String> fields = new ArrayList<String>(m.keySet());
			Collections.sort(fields);
			for (String field : fields) {
				if(!firstTime) {
					r += ",";
				}
				firstTime=false;
				r += field + ":" + toString(m.get(field));
			}
			return r + "}";
		} else if(o != null) {
			// other cases can use their default toString methods.
			return o.toString();
		} else {
			return "null";
		}
	}

	/**
	 * Determine whether a given value is an instanceof a given type. This is
	 * done by recursively exploring the type and the value together, until we
	 * can safely conclude that the value does (or does not) match the required
	 * type.
	 *
	 * @param value
	 * @param type
	 * @return
	 */
	private boolean instanceOf(Object value, Type type) {
		if(type instanceof Type.Void) {
			return false;
		}else if (type instanceof Type.Null) {
			return value == null;
		}else if(type instanceof Type.Bool) {
			return value instanceof Boolean;
		} else if(type instanceof Type.Char) {
			return value instanceof Character;
		} else if(type instanceof Type.Int) {
			return value instanceof Integer;
		} else if(type instanceof Type.Real) {
			return value instanceof Double;
		} else if(type instanceof Type.Strung) {
			return value instanceof StringBuffer;
		} else if (type instanceof Type.List) {
			if (value instanceof ArrayList) {
				Type.List lt = (Type.List) type;
				ArrayList al = (ArrayList) value;
				for (Object o : al) {
					if (!instanceOf(o, lt.getElement())) {
						return false;
					}
				}
				return true;
			}
			return false;
		} else if(type instanceof Type.Record) {
			Type.Record ut = (Type.Record) type;
			if(value instanceof HashMap) {
				HashMap m = (HashMap) value;
				for(Map.Entry<String,Type> p : ut.getFields().entrySet()) {
					if (!instanceOf(m.get(p.getKey()), p.getValue())) {
						return false;
					}
				}
				return true;
			}
			return false;
		} else if(type instanceof Type.Named) {
			return instanceOf(value, userTypes.get(type.toString()));
		} else {
			Type.Union ut = (Type.Union) type;
			for (Type bt : ut.getBounds()) {
				if (instanceOf(value, bt)) {
					return true;
				}
			}
			return false;
		}
	}

	/**
	 * The inverse of the above method - given a WyScript type, return
	 * the corresponding java class. Used for type casting.
	 * Returns null if the type is a union - in this case, no
	 * Java cast is required
	 */
	private Class getJavaClass(Type t) {

		if (t instanceof Type.Bool)
			return Boolean.class;

		else if (t instanceof Type.Int)
			return Integer.class;

		else if (t instanceof Type.Real)
			return Double.class;

		else if (t instanceof Type.Strung)
			return StringBuffer.class;

		else if (t instanceof Type.Char)
			return Character.class;

		else if (t instanceof Type.List) {
			//Is there a better way to do this?
			return ArrayList.class;
		}

		else if (t instanceof Type.Record) {
			return HashMap.class;
		}
		else return null;

	}
}
