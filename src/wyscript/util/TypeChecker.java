package wyscript.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import wyscript.error.TypeErrorData;
import wyscript.error.TypeErrorData.ErrorType;
import wyscript.error.TypeErrorHandler;
import wyscript.lang.*;
import wyscript.lang.Type.Reference;
import static wyscript.util.SyntaxError.*;

/**
 * <p>
 * Responsible for ensuring that all types are used appropriately. For example,
 * that we only perform arithmetic operations on arithmetic types; that we only
 * access fields in records guaranteed to have those fields, etc.
 * </p>
 * <p>
 * Additionally, this phase annotates every expression with the type it returns.
 * This information can be used in subsequent phases (e.g. code generation).
 * </p>
 *
 * @author David J. Pearce
 *
 */
public class TypeChecker {

	private String filename;
	private WyscriptFile.FunDecl function;
	private HashMap<String,WyscriptFile.FunDecl> functions;

	private HashMap<String, Type> userTypes;
	private HashMap<String, Type> constants;
	private ArrayList<TypeErrorData> errors;

	public void check(WyscriptFile wf, String filename) {

		this.filename = filename;
		this.functions = new HashMap<String,WyscriptFile.FunDecl>();
		this.userTypes = new HashMap<String, Type>();
		this.constants = new HashMap<String, Type>();
		this.errors = new ArrayList<TypeErrorData>();

		for(WyscriptFile.Decl declaration : wf.declarations) {
			if(declaration instanceof WyscriptFile.FunDecl) {
				WyscriptFile.FunDecl fd = (WyscriptFile.FunDecl) declaration;
				this.functions.put(fd.name(), fd);
			}
			else if (declaration instanceof WyscriptFile.TypeDecl) {
				WyscriptFile.TypeDecl td = (WyscriptFile.TypeDecl) declaration;
				userTypes.put(td.name(), td.type);
			}

			else if (declaration instanceof WyscriptFile.ConstDecl) {
				WyscriptFile.ConstDecl constant = (WyscriptFile.ConstDecl) declaration;
				constants.put(constant.name(), check(constant.constant, constants));
			}
		}

		for(WyscriptFile.Decl declaration : wf.declarations) {
			if(declaration instanceof WyscriptFile.FunDecl) {
				check((WyscriptFile.FunDecl) declaration);
			}
		}
		if (!errors.isEmpty())
			TypeErrorHandler.handle(errors, userTypes);
	}

	public void check(WyscriptFile.FunDecl fd) {
		this.function = fd;

		// First, initialise the typing environment
		HashMap<String,Type> environment = new HashMap<String,Type>();
		for (WyscriptFile.Parameter p : fd.parameters) {
			environment.put(p.name(), p.type);
		}
		for (String s : constants.keySet()) {
			environment.put(s, constants.get(s));
		}

		// Second, check all statements in the function body
		check(fd.statements,environment);
	}

	public void check(List<Stmt> statements, Map<String,Type> environment) {
		for(Stmt s : statements) {
			check(s,environment);
		}
	}

	public void check(Stmt stmt, Map<String,Type> environment) {
		if(stmt instanceof Stmt.Assign) {
			check((Stmt.Assign) stmt, environment);
		} else if(stmt instanceof Stmt.Print) {
			check((Stmt.Print) stmt, environment);
		} else if(stmt instanceof Stmt.Return) {
			check((Stmt.Return) stmt, environment);
		} else if(stmt instanceof Stmt.VariableDeclaration) {
			check((Stmt.VariableDeclaration) stmt, environment);
		} else if(stmt instanceof Expr.Invoke) {
			check((Expr.Invoke) stmt, environment);
		} else if(stmt instanceof Stmt.IfElse) {
			check((Stmt.IfElse) stmt, environment);
		} else if(stmt instanceof Stmt.OldFor) {
			check((Stmt.OldFor) stmt, environment);
		} else if(stmt instanceof Stmt.For) {
			check((Stmt.For) stmt, environment);
		} else if(stmt instanceof Stmt.While) {
			check((Stmt.While) stmt, environment);
		} else if(stmt instanceof Stmt.Switch) {
			check((Stmt.Switch) stmt, environment);
		} else if(stmt instanceof Stmt.Next) {
			check((Stmt.Next) stmt, environment);
		} else {
			internalFailure("unknown statement encountered (" + stmt + ")", filename,stmt);
		}
	}

	public void check(Stmt.Assign stmt, Map<String,Type> environment) {
		Type lhs = check(stmt.getLhs(), environment);
		Type rhs = check(stmt.getRhs(), environment);
		checkSubtype(lhs,rhs, false, stmt);
		if (stmt.getLhs() instanceof Expr.Tuple) {
			for (Expr e : ((Expr.Tuple)stmt.getLhs()).getExprs()) {
				if (!(e instanceof Expr.LVal))
					errors.add(new TypeErrorData(filename, stmt.getLhs(), e,
							lhs.attribute(Attribute.Source.class), ErrorType.BAD_TUPLE_ASSIGN));
			}
		}
	}

	public void check(Stmt.Print stmt, Map<String,Type> environment) {
		check(stmt.getExpr(),environment);
	}

	public void check(Stmt.Return stmt, Map<String, Type> environment) {
		Expr temp = stmt.getExpr();
		//Check if not returning anything
		if (temp == null && !(function.ret instanceof Type.Void)) {
			Attribute.Source source = function.attribute(Attribute.Source.class);
			source = new Attribute.Source(source.end, source.end);
			errors.add(new TypeErrorData(filename, null, function, source, ErrorType.MISSING_RETURN));
		}
		else if (temp == null)
			return;

		Type actual = check(stmt.getExpr(), environment);
		checkSubtype(function.ret, actual, false, stmt.getExpr());
	}

	public void check(Stmt.VariableDeclaration stmt, Map<String,Type> environment) {
		if(environment.containsKey(stmt.getName())) {
			errors.add(new TypeErrorData(filename, null, stmt,
					stmt.attribute(Attribute.Source.class), ErrorType.DUPLICATE_VARIABLE));
		} else if(stmt.getExpr() != null) {
			Type type = check(stmt.getExpr(),environment);
			checkSubtype(stmt.getType(),type, false, stmt.getExpr());
		}
		environment.put(stmt.getName(), stmt.getType());
	}

	public void check(Stmt.IfElse stmt, Map<String,Type> environment) {
		Type condition = check(stmt.getCondition(),environment);
		checkSubtype(new Type.Bool(), condition, false, stmt.getCondition());
		check(stmt.getTrueBranch(), new HashMap<String,Type>(environment));

		//Check else-if branches
		for (Expr e : stmt.getAltExpressions()) {
			checkSubtype(new Type.Bool(), check(e, environment), false, e);
			check(stmt.getAltBranch(e), new HashMap<String, Type>(environment));
		}

		check(stmt.getFalseBranch(), new HashMap<String,Type>(environment));
	}

	public void check(Stmt.OldFor stmt, Map<String,Type> environment) {
		Stmt.VariableDeclaration d = stmt.getDeclaration();
		if (d != null)
			check(d, environment);

		Expr e = stmt.getCondition();
		if (e != null) {
			Type e_t = check(e, environment);
			checkSubtype(new Type.Bool(), e_t, false, e);
		}

		Stmt s = stmt.getIncrement();
		if (s != null) {
			check(s, environment);
		}
		check(stmt.getBody(), new HashMap<String,Type>(environment));
	}

	public void check(Stmt.For stmt, Map<String, Type> environment) {
		Expr e = stmt.getSource();
		Expr.Variable v = stmt.getIndex();

		Type t = check(e, environment);
		if (!(t instanceof Type.List)) {
			errors.add(new TypeErrorData(filename, stmt.getSource(), null,
					stmt.getSource().attribute(Attribute.Source.class), ErrorType.BAD_FOR_LIST));
			return;
		}

		HashMap<String, Type> newEnv = new HashMap<String, Type>(environment);
		newEnv.put(v.getName(), ((Type.List)t).getElement());
		check(stmt.getBody(), newEnv);
	}

	public void check(Stmt.While stmt, Map<String,Type> environment) {
		Type condition = check(stmt.getCondition(),environment);
		checkSubtype(new Type.Bool(), condition, false, stmt.getCondition());
		check(stmt.getBody(), new HashMap<String,Type>(environment));
	}

	public void check(Stmt.Switch stmt, Map<String, Type> environment) {
		Type expr = check(stmt.getExpr(), environment);

		if (expr instanceof Type.Record || expr instanceof Type.Reference || expr instanceof Type.Tuple) {
			errors.add(new TypeErrorData(filename, stmt.getExpr(), null,
					stmt.getExpr().attribute(Attribute.Source.class), ErrorType.BAD_SWITCH_TYPE));
		}

		for (Stmt.SwitchStmt s : stmt.cases()) {
			if (s instanceof Stmt.Case)
				check((Stmt.Case)s, expr, environment);
			else
				check((Stmt.Default)s, environment);
		}
	}

	public void check(Stmt.Case stmt, Type type, Map<String, Type> environment) {

		checkSubtype(type, check(stmt.getConstant(), environment), false, stmt.getConstant());

		//Workaround to ensure next statements only work within an enclosing case body
		for (Stmt s : stmt.getStmts()) {
			environment.put("#case", new Type.Void());
			check(s, environment);
		}
		environment.remove("#case");
	}

	public void check(Stmt.Default stmt, Map<String, Type> environment) {

		//Workaround to ensure next statements only work within an enclosing case body
		for (Stmt s : stmt.getStmts()) {
			environment.put("#case", new Type.Void());
			check(s, environment);
		}
		environment.remove("#case");
	}

	public void check(Stmt.Next stmt, Map<String, Type> environment) {
		//Check statement is within a case body
		if (environment.get("#case") == null)
			errors.add(new TypeErrorData(filename, null, null,
					stmt.attribute(Attribute.Source.class), ErrorType.BAD_NEXT));
	}

	public Type check(Expr expr, Map<String,Type> environment) {
		Type type;

		if(expr instanceof Expr.Binary) {
			type = check((Expr.Binary) expr, environment);
		} else if(expr instanceof Expr.Cast) {
			type = check((Expr.Cast) expr, environment);
		} else if(expr instanceof Expr.Constant) {
			type = check((Expr.Constant) expr, environment);
		} else if(expr instanceof Expr.IndexOf) {
			type = check((Expr.IndexOf) expr, environment);
		} else if(expr instanceof Expr.Invoke) {
			type = check((Expr.Invoke) expr, environment);
		} else if(expr instanceof Expr.ListConstructor) {
			type = check((Expr.ListConstructor) expr, environment);
		} else if(expr instanceof Expr.RecordAccess) {
			type = check((Expr.RecordAccess) expr, environment);
		} else if(expr instanceof Expr.RecordConstructor) {
			type = check((Expr.RecordConstructor) expr, environment);
		} else if(expr instanceof Expr.Unary) {
			type = check((Expr.Unary) expr, environment);
		} else if(expr instanceof Expr.Variable) {
			type = check((Expr.Variable) expr, environment);
		} else if(expr instanceof Expr.Is) {
			type = check((Expr.Is) expr, environment);
		} else if(expr instanceof Expr.Deref) {
			type = check((Expr.Deref) expr, environment);
		} else if(expr instanceof Expr.New) {
			type = check((Expr.New) expr, environment);
		} else if(expr instanceof Expr.Tuple) {
			type = check((Expr.Tuple) expr, environment);
		} else {
			internalFailure("unknown expression encountered (" + expr + ")", filename,expr);
			return null; // dead code
		}

		// Here, we annotate the computed return type to the expression.
		expr.attributes().add(new Attribute.Type(type));

		return type;
	}

	public Type check(Expr.Binary expr, Map<String,Type> environment) {
		int errCount = errors.size();
		switch (expr.getOp()) {

		case APPEND:
			//Check if lhs is a string, or both types are lists
			boolean isString = false;
			Type left = check(expr.getLhs(), environment);
			Type right = check(expr.getRhs(), environment);

			isString = checkPossibleSubtype(new Type.Strung(), left, false);

			if (!isString) {
				checkSubtype(Type.List.class, left, expr.getLhs());
				checkSubtype(Type.List.class, right, expr.getRhs());
				if (errCount < errors.size())
					return new Type.Void();

				//Check that the RHS is a subtype of the LHS
				checkSubtype(left, right, false, expr.getRhs());
			}
			if (errCount < errors.size())
				return new Type.Void();
			return left;

		case AND:
		case OR:
			checkSubtype(new Type.Bool(), check(expr.getLhs(), environment), false, expr.getLhs());
			checkSubtype(new Type.Bool(), check(expr.getRhs(), environment), false, expr.getRhs());
			return new Type.Bool();

		case EQ:
		case NEQ:
			check(expr.getLhs(), environment);
			check(expr.getRhs(), environment);
			return new Type.Bool();

		case GT:
		case GTEQ:
		case LT:
		case LTEQ:
			Type lhs = check(expr.getLhs(), environment);
			Type rhs = check(expr.getRhs(), environment);

			if (!checkPossibleSubtype(new Type.Int(), lhs, false)) {
				checkSubtype(new Type.Real(), lhs, false, expr.getLhs());
			}
			if (!checkPossibleSubtype(new Type.Int(), rhs, false)) {
				checkSubtype(new Type.Real(), lhs, false, expr.getRhs());
			}

			return new Type.Bool();

		case RANGE:
			checkSubtype(new Type.Int(), check(expr.getLhs(), environment), false, expr.getLhs());
			checkSubtype(new Type.Int(), check(expr.getLhs(), environment), false, expr.getRhs());
			return new Type.List(new Type.Int());

		case REM:
		case SUB:
		case ADD:
		case DIV:
		case MUL:
			Type lhs2  = check(expr.getLhs(), environment);
			Type rhs2 = check(expr.getRhs(), environment);
			boolean promote = false;

			if (!checkPossibleSubtype(new Type.Int(), lhs2, false)) {
				checkSubtype(new Type.Real(), lhs2, false, expr.getLhs());
				promote = true;
			}
			if (!checkPossibleSubtype(new Type.Int(), rhs2, false)) {
				checkSubtype(new Type.Real(), rhs2, false, expr.getRhs());
			}

			if (errCount < errors.size())
				return new Type.Void();
			return (promote) ? new Type.Real() : new Type.Int();

		default:
			internalFailure("Unknown binary operator encountered", filename, expr);
			return null;
		}
	}

	public Type check(Expr.Is expr, Map<String,Type> environment) {
		//Like == or !=, there's no way to have a 'bad' is expression
		//However, we still want to annotate the expression with its type

		check(expr.getLhs(), environment);
		return new Type.Bool();
	}

	public Type check(Expr.Cast expr, Map<String,Type> environment) {

		checkSubtype(check(expr.getSource(), environment), expr.getType(), true, expr.getSource());
		return expr.getType();
	}

	public Type check(Expr.Constant expr, Map<String,Type> environment) {
		Object constant = expr.getValue();

		if(constant instanceof Boolean) {
			return new Type.Bool();
		} else if(constant instanceof Character) {
			return new Type.Char();
		} else if(constant instanceof Integer) {
			return new Type.Int();
		} else if(constant instanceof Double) {
			return new Type.Real();
		} else if(constant instanceof StringBuffer) {
			return new Type.Strung();
		} else if(constant == null) {
			return new Type.Null();
		} else {
			internalFailure("unknown constant encountered (" + expr + ")", filename,expr);
			return null; // dead code
		}
	}

	public Type check(Expr.IndexOf expr, Map<String, Type> environment) {
		Type srcType = check(expr.getSource(), environment);
		Type indexType = check(expr.getIndex(), environment);
		checkSubtype(new Type.Int(), indexType, false, expr.getIndex());

		//Check not indexing a String
		if (checkPossibleSubtype(new Type.Strung(), srcType, false))
			return new Type.Char();

		return checkSubtype(Type.List.class, srcType, expr.getSource())
				.getElement();
	}

	public Type check(Expr.Invoke expr, Map<String,Type> environment) {
		WyscriptFile.FunDecl fn = functions.get(expr.getName());
		List<Expr> arguments = expr.getArguments();
		List<WyscriptFile.Parameter> parameters = fn.parameters;
		if(arguments.size() != parameters.size()) {
			errors.add(new TypeErrorData(filename, expr, fn,
					expr.attribute(Attribute.Source.class), ErrorType.BAD_FUNC_PARAMS));
			return new Type.Void();
		}
		for(int i=0;i!=parameters.size();++i) {
			Type argument = check(arguments.get(i),environment);
			Type parameter = parameters.get(i).type;
			checkSubtype(parameter,argument, false, parameters.get(i));
		}
		return fn.ret;
	}

	public Type check(Expr.ListConstructor expr, Map<String,Type> environment) {
		List<Expr> args = expr.getArguments();
		if (args.isEmpty())
			return new Type.List(new Type.Void());
		//Take the highest possible supertype, or the union of the types, of all the elements
		else {
			List<Type> bounds = new ArrayList<Type>();
			Type t = check(args.get(0), environment);
			for (Expr e : args) {
				Type t2 = check(e, environment);
				if (!checkPossibleSubtype(t, t2, false)) {
					if (checkPossibleSubtype(t2, t, false))
						t = t2;
					else {
						bounds.add(t);
						bounds.add(t2);
						t = new Type.Union(bounds);
					}
				}
			}
			return new Type.List(t);
		}
	}

	public Type check(Expr.RecordAccess expr, Map<String,Type> environment) {

		Type t = check(expr.getSource(), environment);
		Type.Record r = checkSubtype(Type.Record.class, t, expr.getSource());
		if (r == null) {
			errors.add(new TypeErrorData(filename, expr, null,
					expr.attribute(Attribute.Source.class), ErrorType.BAD_FIELD_ACCESS));
			return new Type.Void();
		}

		Type result = r.getFields().get(expr.getName());
		if (result == null) {
			errors.add(new TypeErrorData(filename, expr, null,
					expr.attribute(Attribute.Source.class), ErrorType.MISSING_FIELD));
			return new Type.Void();
		}

		return result;
	}

	public Type check(Expr.RecordConstructor expr, Map<String,Type> environment) {

		Map<String, Type> fields = new HashMap<String, Type>();

		for (Pair<String, Expr> p : expr.getFields()) {
			fields.put(p.first(), check(p.second(), environment));
		}
		return new Type.Record(fields);
	}

	public Type check(Expr.Unary expr, Map<String,Type> environment) {

		int errCount = errors.size();
		switch(expr.getOp()) {

		case LENGTHOF:
			if (!checkPossibleSubtype(new Type.Strung(), check(expr.getExpr(), environment), false)) {
				checkSubtype(Type.List.class, check(expr.getExpr(), environment), expr.getExpr());
			}
			return new Type.Int();

		case NEG:
			Type t = check(expr.getExpr(), environment);

			if (!checkPossibleSubtype(new Type.Int(), t, false)) {
				checkSubtype(new Type.Real(), t, false, expr.getExpr());
				return new Type.Real();
			}
			if (errCount < errors.size())
				return new Type.Void();
			return new Type.Int();

		case NOT:
			checkSubtype(new Type.Bool(), check(expr.getExpr(), environment), false, expr.getExpr());
			return new Type.Bool();

		default:
			internalFailure("Unknown unary operator encountered", filename, expr);
			return null;
		}
	}

	public Type check(Expr.Variable expr, Map<String, Type> environment) {
		Type type = environment.get(expr.getName());
		if (type == null) {
			errors.add(new TypeErrorData(filename, expr, null, expr.attribute(Attribute.Source.class), ErrorType.UNDECLARED_VARIABLE));
			return new Type.Void();
		}
		return type;
	}

	public Type check(Expr.Deref expr, Map<String, Type> environment) {
		Type type = check(expr.getExpr(), environment);
		return ((Type.Reference)type).getType();
	}

	public Type check(Expr.New expr, Map<String, Type> environment) {
		Type type = check(expr.getExpr(), environment);
		return new Type.Reference(type);
	}

	public Type check(Expr.Tuple expr, Map<String, Type> environment) {
		List<Type> types = new ArrayList<Type>();

		for (Expr e : expr.getExprs())
			types.add(check(e, environment));

		return new Type.Tuple(types);
	}

	/**
	 * Check that a given type t2 is an instance of of another type t1. This
	 * method is useful for checking that a type is, for example, a List type.
	 *
	 * @param t1
	 * @param t2
	 * @param element
	 *            Used for determining where to report syntax errors.
	 * @return
	 */
	public <T extends Type> T checkSubtype(Class<T> t1, Type t2,
			Expr element) {
		if (t1.isInstance(t2)) {
			return (T) t2;
		}
		else if (t2 instanceof Type.Named)
			return (checkSubtype(t1, userTypes.get(((Type.Named)t2).getName()), element));
		else {

			//Must avoid double-reporting errors
			if (!t1.equals(Type.Record.class)) {
				Type t = getTypeFromClass(t1);
					errors.add(new TypeErrorData(filename, element, t,
						element.attribute(Attribute.Source.class), ErrorType.TYPE_MISMATCH));
			}

			return null;
		}
	}

	/**
	 * Check that a given type t2 is a subtype of another type t1.
	 *
	 * @param t1 Supertype to check
	 * @param t2 Subtype to check
	 * @param element
	 *            Used for determining where to report syntax errors.
	 */
	public void checkSubtype(Type t1, Type t2, boolean cast, SyntacticElement element) {

		if (checkPossibleSubtype(t1, t2, cast)) {
			//OK!
		}
		else {
			element.attributes().add(new Attribute.Type(t2));
			Expr e = new Expr.Cast(t1, null);
			errors.add(new TypeErrorData(filename, e, element,
					element.attribute(Attribute.Source.class), ErrorType.SUBTYPE_MISMATCH));
		}
	}

	/**
	 * Utility method - checks if a type is a subtype of another type, returning
	 * the result of that check.
	 *
	 * @param t1 - The supertype
	 * @param t2 - The (possible) subtype being checked
	 */
	private boolean checkPossibleSubtype(Type t1, Type t2, boolean cast) {

		//Attempt to normalize record types
		if (t1 instanceof Type.Record || t2 instanceof Type.Record) {

			if (t1 instanceof Type.Record) {
				Attribute type = normalize((Type.Record) t1);
				t1.attributes().add(type);
			}
			if (t2 instanceof Type.Record) {
				Attribute type = normalize((Type.Record) t2);
				t2.attributes().add(type);
			}
		}

		//Attempt to normalize tuple types
		if (t1 instanceof Type.Tuple || t2 instanceof Type.Tuple) {
			if (t1 instanceof Type.Tuple) {
				Attribute type = normalize((Type.Tuple) t1);
				t1.attributes().add(type);
			}
			if (t2 instanceof Type.Tuple) {
				Attribute type = normalize((Type.Tuple) t2);
				t2.attributes().add(type);
			}
		}


		if (t1 instanceof Type.Bool && t2 instanceof Type.Bool) {
			return true;
		} else if (t1 instanceof Type.Char && t2 instanceof Type.Char) {
			return true;
		} else if (t1 instanceof Type.Int && t2 instanceof Type.Int) {
			return true;
		} else if (t1 instanceof Type.Real && t2 instanceof Type.Real) {
			return true;
		} else if (t1 instanceof Type.Strung && t2 instanceof Type.Strung) {
			return true;
		} else if (t1 instanceof Type.Void && t2 instanceof Type.Void) {
			return true;
		} else if (t1 instanceof Type.Null && t2 instanceof Type.Null) {
			return true;
		} else if (cast && t1 instanceof Type.Real && t2 instanceof Type.Int) {
			return true;
		} else if (cast && t1 instanceof Type.Int && t2 instanceof Type.Real) {
			return true;
		}

		else if (t1 instanceof Type.List && t2 instanceof Type.List) {
			Type.List l1 = (Type.List) t1;
			Type.List l2 = (Type.List) t2;
			// The following is safe because While has value semantics. In a
			// conventional language, like Java, this is not safe because of
			// references.
			return checkPossibleSubtype(l1.getElement(),l2.getElement(), cast);
		}
		//Records implement depth subtyping, but not width subtyping. Thus a record is
		//a subtype of another record if it has the same number of fields, those
		//fields have the same names, and the types of t2's fields are subtypes of the
		//types of t1's fields
		else if (t1 instanceof Type.Record && t2 instanceof Type.Record) {

			Type.Record r1 = (Type.Record) t1;
			Type.Record r2 = (Type.Record) t2;
			Map<String, Type> f1 = r1.getFields();
			Map<String, Type> f2 = r2.getFields();

			if (f1.size() != f2.size())
				return false;

			for (String s : f1.keySet()) {
				if (f2.get(s) == null)
					return false;

				if (checkPossibleSubtype(f1.get(s), f2.get(s), cast) == false)
					return false;
			}
			return true;
		}
		else if (t1 instanceof Type.Tuple && t2 instanceof Type.Tuple) {
			Type.Tuple tup1 = (Type.Tuple) t1;
			Type.Tuple tup2 = (Type.Tuple) t2;

			if (tup1.getTypes().size() != tup2.getTypes().size())
				return false;

			for (int i = 0; i < tup1.getTypes().size(); i++) {
				if (!(checkPossibleSubtype(tup1.getTypes().get(i), tup2.getTypes().get(i), cast)))
					return false;
			}
			return true;
		}
		//A union is a subtype of a union if all its bounds are subtypes of t1's bounds
		else if (t1 instanceof Type.Union && t2 instanceof Type.Union) {

			Type.Union u1 = (Type.Union) t1;
			Type.Union u2 = (Type.Union) t2;

			for (Type ut2 : u2.getBounds()) {
				boolean subtype = false;
				for (Type ut1 : u1.getBounds()) {
					if (checkPossibleSubtype(ut1, ut2, cast)) {
						subtype = true;
						break;
					}
				}
				if (!subtype)
					return false;
			}
			return true;
		}
		//A union is a subtype of a type if all of its bounds are within the type
		else if (t2 instanceof Type.Union) {
			Type.Union u2 = (Type.Union) t2;
			for (Type t : u2.getBounds()) {
				if (!checkPossibleSubtype(t1, t, cast)) {
					return false;
				}
			}
			return true;
		}
		//A type is a subtype of a union if it is a subtype of any of the union's bounds
		else if (t1 instanceof Type.Union) {
			Type.Union u1 = (Type.Union) t1;
			boolean subtype = false;
			Type actual = t2;
			//Handle normalized types
			if (t2 instanceof Type.Record || t2 instanceof Type.Tuple) {
				actual = t2.attribute(Attribute.Type.class).type;
			}
			for (Type t : u1.getBounds()) {
				if (checkPossibleSubtype(t, actual, cast)) {
					subtype = true;
					break;
				}
			}
			return subtype;
		}
		//When checking named types, need to convert to actual type
		else if (t1 instanceof Type.Named || t2 instanceof Type.Named) {
			Type tt1, tt2;

			if (t1 instanceof Type.Named) {
				tt1 = userTypes.get(((Type.Named)t1).getName());
				if (tt1 == null)
					internalFailure("Error, couldn't find type associated with " + t1, filename, tt1);
			}
			else tt1 = t1;

			if (t2 instanceof Type.Named) {
				tt2 = userTypes.get(((Type.Named)t2).getName());
				if (tt2 == null)
					internalFailure("Error, couldn't find type associated with " + t2, filename, t2);
			}
			else tt2 = t2;

			return checkPossibleSubtype(tt1, tt2, cast);
		}
		//void is a subtype of every type
		else if (t2 instanceof Type.Void) {
			return true;
		}

		else if (t1 instanceof Type.Reference && t2 instanceof Type.Reference) {
			Type newT1 = ((Type.Reference)t1).getType();
			Type newT2 = ((Type.Reference)t2).getType();

			return checkPossibleSubtype(newT1, newT2, cast);
		} else return false;
	}

	/**
	 * Attempts to normalize the type of a Tuple - if the tuple contains
	 * any union types (eg (int|null, int) ), those types are extracted
	 * to create a union type of tuples ( (int, int) | (null, int) )
	 */
	private Attribute normalize(Type.Tuple t) {

		List<Type> types = t.getTypes();

		boolean hasUnion = false;
		for (Type type : types) {
			if (type instanceof Type.Union)
				hasUnion = true;
		}

		return (hasUnion) ? getTupleUnion(t) : new Attribute.Type(t);
	}

	/**
	 * Wrapper Method - Calls getTupleUnion to get the bounds
	 * of the resulting union type, then wraps that union in
	 * an attribute and returns it
	 * @param types
	 * @return
	 */
	private Attribute getTupleUnion(Type.Tuple t) {

		List<List<Type>> tupleTypes = getTupleUnion(t.getTypes());

		List<Type> bounds = new ArrayList<Type>();

		for (List<Type> typeList : tupleTypes)
			bounds.add(new Type.Tuple(typeList));

		return new Attribute.Type(new Type.Union(bounds));
	}

	/**
	 * Takes a list of types and returns a list of all the possible type lists
	 * after normalisation.
	 */
	private List<List<Type>> getTupleUnion(List<Type> types) {

		List<List<Type>> result = new ArrayList<List<Type>>();

		//First, split the type list up into the first element and the remainder,
		//and get the normalised typelist for the remainder (getting nothing if the typelist only
		//had one element)
		Type elem = types.get(0);
		List<List<Type>> other = (types.size() > 1) ? getTupleUnion(types.subList(1, types.size()-1))
													: new ArrayList<List<Type>>();
		List<Type> current = new ArrayList<Type>();

		//Create a list of all the types for the current element
		if (elem instanceof Type.Union) {
			for (Type t : ((Type.Union)elem).getBounds()) {
				current.add(t);
			}
		}
		else current.add(elem);

		//For all the union types we've built up so far,
		//create a new list(s) with the current type(s) at the front
		for (List<Type> lt : other) {
			for (Type t : current) {
				List<Type> tmp = new ArrayList<Type>(lt);
				tmp.add(0, t);
				result.add(tmp);
			}
		}
		//The case where the type list only contained one element
		if (result.isEmpty()) {
			for (Type t : current) {
				List<Type> tmp = new ArrayList<Type>();
				tmp.add(t);
				result.add(tmp);
			}
		}

		return result;
	}

	/**
	 * Attempts to normalize the type of a Record - if the record contains
	 * any fields of union type (eg int|null x), those types are extracted
	 * to create a union type of records ( {int x} | {null x} )
	 */
	private Attribute normalize(Type.Record r) {

		//The set of all possible types each field can have
		Map<String, Set<Type>> types = new HashMap<String, Set<Type>>();

		for (String s : r.getFields().keySet()) {

			Set<Type> fieldTypes = new HashSet<Type>();

			Type fieldType = r.getFields().get(s);
			if (fieldType instanceof Type.Union) {
				for (Type t : ((Type.Union)fieldType).getBounds())
					fieldTypes.add(t);
			}
			else fieldTypes.add(fieldType);
			types.put(s, fieldTypes);
		}

		boolean isUnion = false;
		for (String s : r.getFields().keySet()) {
			Set<Type> typeSet = types.get(s);
			if (typeSet.size() > 0)
				isUnion = true;
			break;
		}
		return (isUnion) ? new Attribute.Type(getRecordUnion(types)) : new Attribute.Type(r);
	}

	/**
	 * Wrapper method - calls getRecordPerms to get the set of all possible records,
	 * then converts that set into a union type and returns it.
	 */
	private Type.Union getRecordUnion(Map<String, Set<Type>> types) {
		//Need to convert the map to a set of pairs of <String, Set<Type>>

		Set<Pair<String, Set<Type>>> typeSet = new HashSet<Pair<String,Set<Type>>>();

		for (String s : types.keySet()) {
			typeSet.add(new Pair<String, Set<Type>>(s, types.get(s)));
		}
		//Now pass that set along to the recursive method and get the set of all possible records back
		Set<Map<String, Type>> records = getRecordPerms(typeSet);

		List<Type> bounds = new ArrayList<Type>();

		for (Map<String, Type> field : records) {
			bounds.add(new Type.Record(field));
		}
		return new Type.Union(bounds);
	}

	/**
	 * Recursive method that builds up the set of all possible fields a record could have.
	 * Takes a set containing all field names and the set of types of each field, and splits
	 * that into two - the first element and the remainder. It then passed the remainder on
	 * to getRecordPerms() (if the remainder exists) to get a set of half-built field maps.
	 * Then it iterates through all the types attached to the first element, creating a (potentially)
	 * larger set of field-maps, with one more entry in each map.
	 */
	private Set<Map<String, Type>> getRecordPerms(Set<Pair<String, Set<Type>>> typeSet) {

		Set<Pair<String, Set<Type>>> remainder = new HashSet<Pair<String, Set<Type>>>();

		boolean first = true;
		Pair<String, Set<Type>> current = null;

		for (Pair<String, Set<Type>> pair : typeSet) {
			if (first) {
				current = pair;
				first = false;
			}
			else remainder.add(pair);
		}

		Set<Map<String, Type>> tmp = null;
		if (!remainder.isEmpty()) {
			tmp = getRecordPerms(remainder);
		}

		Set<Map<String, Type>> result = new HashSet<Map<String, Type>>();

		for (Type t : current.second()) {
			if (tmp == null) {
				Map<String, Type> m = new HashMap<String, Type>();
				m.put(current.first(), t);
				result.add(m);
			}
			else {
				for (Map<String, Type> fields : tmp) {
					Map<String, Type> m = new HashMap<String, Type>(fields);
					m.put(current.first(), t);
					result.add(m);
				}
			}
		}

		return result;
	}

	/**
	 * Gets a generic type, given the class of that type.
	 * Guaranteed to only be called for record or list types
	 */
	private <T extends Type> Type getTypeFromClass(Class<T> t1) {
		if (t1.equals(Type.List.class))
			return new Type.List(new Type.Void());
		else return new Type.Record(new HashMap<String, Type>());

	}
}
