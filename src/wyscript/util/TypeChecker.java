package wyscript.util;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import wyscript.lang.*;
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
	private WyscriptFile file;
	private WyscriptFile.FunDecl function;
	private HashMap<String,WyscriptFile.FunDecl> functions;

	public void check(WyscriptFile wf) {
		this.file = wf;
		this.functions = new HashMap<String,WyscriptFile.FunDecl>();

		for(WyscriptFile.Decl declaration : wf.declarations) {
			if(declaration instanceof WyscriptFile.FunDecl) {
				WyscriptFile.FunDecl fd = (WyscriptFile.FunDecl) declaration;
				this.functions.put(fd.name(), fd);
			}
		}

		for(WyscriptFile.Decl declaration : wf.declarations) {
			if(declaration instanceof WyscriptFile.FunDecl) {
				check((WyscriptFile.FunDecl) declaration);
			}
		}
	}

	public void check(WyscriptFile.FunDecl fd) {
		this.function = fd;

		// First, initialise the typing environment
		HashMap<String,Type> environment = new HashMap<String,Type>();
		for (WyscriptFile.Parameter p : fd.parameters) {
			environment.put(p.name(), p.type);
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
		} else if(stmt instanceof Stmt.While) {
			check((Stmt.While) stmt, environment);
		} else {
			internalFailure("unknown statement encountered (" + stmt + ")", file.filename,stmt);
		}
	}

	public void check(Stmt.Assign stmt, Map<String,Type> environment) {
		Type lhs = check(stmt.getLhs(), environment);
		Type rhs = check(stmt.getRhs(), environment);
		checkSubtype(lhs,rhs,stmt);
	}

	public void check(Stmt.Print stmt, Map<String,Type> environment) {
		check(stmt.getExpr(),environment);
	}

	public void check(Stmt.Return stmt, Map<String, Type> environment) {
		Type actual = check(stmt.getExpr(), environment);
		checkSubtype(function.ret, actual, stmt.getExpr());
	}

	public void check(Stmt.VariableDeclaration stmt, Map<String,Type> environment) {
		if(environment.containsKey(stmt.getName())) {
			syntaxError("variable already declared: " + stmt.getName(),
					file.filename, stmt);
		} else if(stmt.getExpr() != null) {
			Type type = check(stmt.getExpr(),environment);
			checkSubtype(stmt.getType(),type,stmt);
		}
		environment.put(stmt.getName(), stmt.getType());
	}

	public void check(Stmt.IfElse stmt, Map<String,Type> environment) {
		Type condition = check(stmt.getCondition(),environment);
		checkSubtype(Type.Bool.class, condition, stmt.getCondition());
		check(stmt.getTrueBranch(), new HashMap<String,Type>(environment));
		check(stmt.getFalseBranch(), new HashMap<String,Type>(environment));
	}

	public void check(Stmt.OldFor stmt, Map<String,Type> environment) {
		// TODO: implement me!
	}

	public void check(Stmt.While stmt, Map<String,Type> environment) {
		Type condition = check(stmt.getCondition(),environment);
		checkSubtype(Type.Bool.class, condition, stmt.getCondition());
		check(stmt.getBody(), new HashMap<String,Type>(environment));
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
			type = check((Expr.ListConstructor) expr, environment);
		} else if(expr instanceof Expr.Unary) {
			type = check((Expr.Unary) expr, environment);
		} else if(expr instanceof Expr.Variable) {
			type = check((Expr.Variable) expr, environment);
		} else {
			internalFailure("unknown expression encountered (" + expr + ")", file.filename,expr);
			return null; // dead code
		}

		// Here, we annotate the computed return type to the expression.
		expr.attributes().add(new Attribute.Type(type));

		return type;
	}

	public Type check(Expr.Binary expr, Map<String,Type> environment) {
		// TODO: implement me
		return null;
	}

	public Type check(Expr.Cast expr, Map<String,Type> environment) {
		// TODO: implement me
		return null;
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
		} else if(constant instanceof String) {
			return new Type.Strung();
		} else if(constant == null) {
			return new Type.Null();
		} else {
			internalFailure("unknown constant encountered (" + expr + ")", file.filename,expr);
			return null; // dead code
		}
	}

	public Type check(Expr.IndexOf expr, Map<String, Type> environment) {
		Type srcType = check(expr.getSource(), environment);
		Type indexType = check(expr.getIndex(), environment);
		checkSubtype(Type.Int.class, indexType, expr.getIndex());
		return checkSubtype(Type.List.class, srcType, expr.getSource())
				.getElement();
	}

	public Type check(Expr.Invoke expr, Map<String,Type> environment) {
		WyscriptFile.FunDecl fn = functions.get(expr.getName());
		List<Expr> arguments = expr.getArguments();
		List<WyscriptFile.Parameter> parameters = fn.parameters;
		if(arguments.size() != parameters.size()) {
			syntaxError("incorrect number of arguments to function",
					file.filename, expr);
		}
		for(int i=0;i!=parameters.size();++i) {
			Type argument = check(arguments.get(i),environment);
			Type parameter = parameters.get(i).type;
			checkSubtype(parameter,argument,parameters.get(i));
		}
		return fn.ret;
	}

	public Type check(Expr.ListConstructor expr, Map<String,Type> environment) {
		// TODO: implement me
		return null;
	}

	public Type check(Expr.RecordAccess expr, Map<String,Type> environment) {
		// TODO: implement me
		return null;
	}

	public Type check(Expr.RecordConstructor expr, Map<String,Type> environment) {
		// TODO: implement me
		return null;
	}

	public Type check(Expr.Unary expr, Map<String,Type> environment) {
		// TODO: implement me
		return null;
	}

	public Type check(Expr.Variable expr, Map<String, Type> environment) {
		Type type = environment.get(expr.getName());
		if (type == null) {
			syntaxError("unknown variable encountered: " + expr.getName(),
					file.filename, expr);
		}
		return type;
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
			SyntacticElement element) {
		if (t1.isInstance(t2)) {
			return (T) t2;
		} else {
			syntaxError("expected instance of " + t1.getClass().getName()
					+ ", found " + t2, file.filename, element);
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
	public void checkSubtype(Type t1, Type t2, SyntacticElement element) {
		if (t1 instanceof Type.Bool && t2 instanceof Type.Bool) {
			// OK
		} else if (t1 instanceof Type.Char && t2 instanceof Type.Char) {
			// OK
		} else if (t1 instanceof Type.Int && t2 instanceof Type.Int) {
			// OK
		} else if (t1 instanceof Type.Real && t2 instanceof Type.Real) {
			// OK
		} else if (t1 instanceof Type.Strung && t2 instanceof Type.Strung) {
			// OK
		} else if (t1 instanceof Type.List && t2 instanceof Type.List) {
			Type.List l1 = (Type.List) t1;
			Type.List l2 = (Type.List) t2;
			// The following is safe because While has value semantics. In a
			// conventional language, like Java, this is not safe because of
			// references.
			checkSubtype(l1.getElement(),l2.getElement(),element);
		} else {
			syntaxError("expected type " + t1
					+ ", found " + t2, file.filename, element);
		}
	}
}
