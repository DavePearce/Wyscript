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

package wyscript.lang;

import java.util.*;

import wyscript.par.KernelRunner;
import wyscript.util.*;
import wyscript.util.SyntaxError.InternalFailure;

/**
 * Represents a statement in the source code of a While program. Many standard
 * statement kinds are provided, including <code>if</code>, <code>while</code>,
 * <code>for</code>, etc.
 *
 * @author David J. Pearce
 *
 */
public interface Stmt extends SyntacticElement {

	/**
	 * Atomic statements are those which are not composed from other statements.
	 * For example, <code>if</code> statements are an example of non-atomic
	 * (compound) statements, whilst assignments are examples of atomic statements.
	 *
	 * @author David J. Pearce
	 *
	 */
	public interface Atom extends Stmt {}

	/**
	 * Represents an assignment statement of the form <code>lhs = rhs</code>.
	 * Here, the <code>rhs</code> is any expression, whilst the <code>lhs</code>
	 * must be an <code>LVal</code> --- that is, an expression permitted on the
	 * left-side of an assignment. The following illustrates different possible
	 * assignment statements:
	 *
	 * <pre>
	 * x = y       // variable assignment
	 * x.f = y     // field assignment
	 * x[i] = y    // list assignment
	 * x[i].f = y  // compound assignment
	 * </pre>
	 *
	 * The last assignment here illustrates that the left-hand side of an
	 * assignment can be arbitrarily complex, involving nested assignments into
	 * lists and records.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Assign extends SyntacticElement.Impl implements
			Atom {

		private final Expr.LVal lhs;
		private final Expr rhs;

		/**
		 * Create an assignment from a given <code>lhs</code> and
		 * <code>rhs</code>.
		 *
		 * @param lhs
		 *            --- left-hand side, which may not be <code>null</code>.
		 * @param rhs
		 *            --- right-hand side, which may not be <code>null</code>.
		 * @param attributes
		 */
		public Assign(Expr.LVal lhs, Expr rhs, Attribute... attributes) {
			super(attributes);
			this.lhs = lhs;
			this.rhs = rhs;
		}

		/**
		 * Create an assignment from a given <code>lhs</code> and
		 * <code>rhs</code>.
		 *
		 * @param lhs
		 *            left-hand side, which may not be <code>null</code>.
		 * @param rhs
		 *            right-hand side, which may not be <code>null</code>.
		 * @param attributes
		 */
		public Assign(Expr.LVal lhs, Expr rhs, Collection<Attribute> attributes) {
			super(attributes);
			this.lhs = lhs;
			this.rhs = rhs;
		}

		public String toString() {
			return getLhs() + " = " + getRhs();
		}

		/**
		 * Get the left-hand side of this assignment.
		 *
		 * @return Guaranteed non-null.
		 */
		public Expr.LVal getLhs() {
			return lhs;
		}

		/**
		 * Get the right-hand side of this assignment.
		 *
		 * @return Guaranteed non-null.
		 */
		public Expr getRhs() {
			return rhs;
		}
	}

	/**
	 * Represents a return statement which (optionally) returns a value. The
	 * following illustrates:
	 *
	 * <pre>
	 * int f(int x):
	 * 	  return x + 1
	 * </pre>
	 *
	 * Here, we see a simple <code>return</code> statement which returns an
	 * <code>int</code> value.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Return extends SyntacticElement.Impl implements
			Atom {

		private final Expr expr;

		/**
		 * Create a given return statement with an optional return value.
		 *
		 * @param expr
		 *            the return value, which may be <code>null</code>.
		 * @param attributes
		 */
		public Return(Expr expr, Attribute... attributes) {
			super(attributes);
			this.expr = expr;
		}

		/**
		 * Create a given return statement with an optional return value.
		 *
		 * @param expr
		 *            the return value, which may be <code>null</code>.
		 * @param attributes
		 */
		public Return(Expr expr, Collection<Attribute> attributes) {
			super(attributes);
			this.expr = expr;
		}

		public String toString() {
			if (getExpr() != null) {
				return "return " + getExpr();
			} else {
				return "return";
			}
		}

		/**
		 * Get the optional return value.
		 *
		 * @return --- May be <code>null</code>.
		 */
		public Expr getExpr() {
			return expr;
		}
	}

	/**
	 * Represents a while statement whose body is made up from a block of
	 * statements separated by curly braces. Note that, unlike C or Java, the
	 * body must be contained within curly braces. As an example:
	 *
	 * <pre>
	 * int sum([int] xs):
	 *   int r = 0
	 *   int i = 0
	 *   while(i < |xs|):
	 *     r = r + xs[i]
	 *     i = i + 1
	 *   return r
	 * </pre>
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class While extends SyntacticElement.Impl implements
			Stmt {

		private final Expr condition;
		private final ArrayList<Stmt> body;

		/**
		 * Construct a While statement from a given condition and body of
		 * statements.
		 *
		 * @param condition
		 *            non-null expression.
		 * @param body
		 *            non-null collection which contains zero or more
		 *            statements.
		 * @param attributes
		 */
		public While(Expr condition, Collection<Stmt> body,
				Attribute... attributes) {
			super(attributes);
			this.condition = condition;
			this.body = new ArrayList<Stmt>(body);
		}

		/**
		 * Construct a While statement from a given condition and body of
		 * statements.
		 *
		 * @param condition
		 *            non-null expression.
		 * @param body
		 *            non-null collection which contains zero or more
		 *            statements.
		 * @param attributes
		 */
		public While(Expr condition, Expr invariant, Collection<Stmt> body,
				Collection<Attribute> attributes) {
			super(attributes);
			this.condition = condition;
			this.body = new ArrayList<Stmt>(body);
		}

		/**
		 * Get the condition which controls the while loop.
		 *
		 * @return Guaranteed to be non-null.
		 */
		public Expr getCondition() {
			return condition;
		}

		/**
		 * Get the statements making up the loop body.
		 *
		 * @return Guarantted to be non-null.
		 */
		public List<Stmt> getBody() {
			return body;
		}
	}

	/**
	 * Represents a foreach statement which iterates a given <i>index
	 * variable</i> over every element of a <i>source expression</i> (which must
	 * return a list). The following illustrates:
	 *
	 * <pre>
	 * int sum([int] xs):
	 *   int r = 0
	 *   for i in xs:
	 *     r = r + xs[i]
	 *   return r
	 * </pre>
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class For extends SyntacticElement.Impl implements Stmt {
		private final Expr.Variable index;
		private final Expr source;
		private final ArrayList<Stmt> body;

		/**
		 * Construct a for loop from a given index variable, source expression
		 * and loop body.
		 *
		 * @param index
		 *            The index variable, which may not be null
		 * @param source
		 *            The source expression, which may not be null
		 * @param body
		 *            A list of zero or more statements, which may not be null.
		 * @param attributes
		 */
		public For(Expr.Variable index, Expr source, Collection<Stmt> body,
				Attribute... attributes) {
			super(attributes);
			this.index = index;
			this.source = source;
			this.body = new ArrayList<Stmt>(body);
		}

		/**
		 * Construct a for loop from a given index variable, source expression
		 * and loop body.
		 *
		 * @param index
		 *            The index variable, which may not be null
		 * @param source
		 *            The source expression, which may not be null
		 * @param body
		 *            A list of zero or more statements, which may not be null.
		 * @param attributes
		 */
		public For(Expr.Variable index, Expr source, Collection<Stmt> body,
				Collection<Attribute> attributes) {
			super(attributes);
			this.index = index;
			this.source = source;
			this.body = new ArrayList<Stmt>(body);
		}

		/**
		 * Get the index variable for this loop.
		 *
		 * @return May not be null.
		 */
		public Expr.Variable getIndex() {
			return index;
		}

		/**
		 * Get the source expression for this loop.
		 *
		 * @return May not be null.
		 */
		public Expr getSource() {
			return source;
		}

		/**
		 * Get the loop body.
		 *
		 * @return May not be null.
		 */
		public ArrayList<Stmt> getBody() {
			return body;
		}
	}

	/**
	 * Represents a classical for statement made up from a <i>variable
	 * declaration</i>, a <i>loop condition</i> and an <i>increment
	 * statement</i>. The following illustrates:
	 *
	 * <pre>
	 * int sum([int] xs) {
	 *   int r = 0;
	 *   for(int i=0;i<|xs|;i=i+1) {
	 *     r = r + xs[i];
	 *   }
	 *   return r;
	 * }
	 * </pre>
	 *
	 * Observe that the variable declaration does not need to supply an
	 * initialiser expression. Furthermore, like C and Java, the variable
	 * declaration, condition and increment statements are all optional. Thus,
	 * we can safely rewrite the above as follows:
	 *
	 * <pre>
	 * int sum([int] xs) {
	 *   int r = 0;
	 *   for(int i=0;i<|xs|;) {
	 *     r = r + xs[i];
	 *   }
	 *   return r;
	 * }
	 * </pre>
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class OldFor extends SyntacticElement.Impl implements Stmt {

		private final VariableDeclaration declaration;
		private final Expr condition;
		private final Stmt increment;
		private final ArrayList<Stmt> body;

		/**
		 * Construct a for loop from a given declaration, condition and
		 * increment. Note that the declaration, conditional and increment are
		 * all optional.
		 *
		 * @param declaration
		 *            An variable declation, which may be null.
		 * @param condition
		 *            A loop condition which may not be null.
		 * @param increment
		 *            An increment statement, which may be null.
		 * @param body
		 *            A list of zero or more statements, which may not be null.
		 * @param attributes
		 */
		public OldFor(VariableDeclaration declaration, Expr condition, Stmt increment,
				Collection<Stmt> body, Attribute... attributes) {
			super(attributes);
			this.declaration = declaration;
			this.condition = condition;
			this.increment = increment;
			this.body = new ArrayList<Stmt>(body);
		}

		/**
		 * Construct a for loop from a given declaration, condition and
		 * increment. Note that the declaration, conditional and increment are
		 * all optional.
		 *
		 * @param declaration
		 *            An variable declation, which may be null.
		 * @param condition
		 *            A loop condition which may be null.
		 * @param increment
		 *            An increment statement, which may be null.
		 * @param body
		 *            A list of zero or more statements, which may not be null.
		 * @param attributes
		 */
		public OldFor(VariableDeclaration declaration, Expr condition, Stmt increment,
				Collection<Stmt> body, Collection<Attribute> attributes) {
			super(attributes);
			this.declaration = declaration;
			this.condition = condition;
			this.increment = increment;
			this.body = new ArrayList<Stmt>(body);
		}

		/**
		 * Get the variable declaration for this loop.
		 *
		 * @return May be null.
		 */
		public VariableDeclaration getDeclaration() {
			return declaration;
		}

		/**
		 * Get the loop condition.
		 *
		 * @return May be null.
		 */
		public Expr getCondition() {
			return condition;
		}

		/**
		 * Get the increment statement.
		 *
		 * @return May be null.
		 */
		public Stmt getIncrement() {
			return increment;
		}

		/**
		 * Get the loop body.
		 *
		 * @return May not be null.
		 */
		public ArrayList<Stmt> getBody() {
			return body;
		}
	}
	public static final class ParFor extends SyntacticElement.Impl implements Stmt {
		private final List<Stmt> body;
		private BoundCalc calc;
		private KernelRunner runner;
		//three indices, index2 and index3 may not be set
		public final Expr.Variable indexX;
		public final Expr.Variable indexY;
		public final Expr.Variable indexZ;
		//each src expression matches to one index
		public final Expr srcX;
		public final Expr srcY;
		public final Expr srcZ;
		//depth of the loop determined by number of 'and's that occur
		public final int depth;
		/**
		 * Construct a for loop from a given index variable, source expression
		 * and loop body.
		 *
		 * @param index
		 *            The index variable, which may not be null
		 * @param source
		 *            The source expression, which may not be null
		 * @param body
		 *            A list of zero or more statements, which may not be null.
		 * @param attributes
		 */
		public ParFor(Expr.Variable index1, Expr.Variable index2, Expr.Variable index3,
				Expr src1, Expr src2 , Expr src3, List<Stmt> body,
				Attribute... attributes) {
			this.indexX = index1;
			this.indexY = index2;
			this.indexZ = index3;
			this.srcX = src1;
			this.srcY = src2;
			this.srcZ = src3;
			this.body = body;
			if (src2 == null) depth = 1;
			else if (src3 == null) depth = 2;
			else depth = 3;
			this.calc = new BoundCalc(this);
		}
		/**
		 * Get the loop body.
		 *
		 * @return May not be null.
		 */
		public List<Stmt> getBody() {
			return body;
		}
		/**
		 * Give the parallel for a kernel runner to execute code on the GPU.
		 * @param runner
		 */
		public void setKernelRunner(KernelRunner runner) {
			this.runner = runner;
		}
		/**
		 * @return The kernel runner associated with this parallel for
		 */
		public KernelRunner getRunner() {
			return this.runner;
		}

		public BoundCalc getCalc() {
			return calc;
		}

		public void setCalc(BoundCalc calc) {
			this.calc = calc;
		}

	}
	/**
	 * Represents a classical if-else statement, made up from a
	 * <i>condition</i>, <i>true branch</i>, optional additional <i> else-if branches </i>
	 * and <i>false branch</i>.
	 * The following illustrates:
	 * <pre>
	 * int max(int x, int y):
	 *   if(x > y):
	 *     return x
	 *   else if(x == y):
	 *   	return 0
	 *   else:
	 *     return y
	 * </pre>
	 * @author David J. Pearce
	 *
	 */
	public static final class IfElse extends SyntacticElement.Impl implements
			Stmt {

		private final Expr condition;
		private final ArrayList<Stmt> trueBranch;
		private final ArrayList<Stmt> falseBranch;
		private final Map<Expr, List<Stmt>> alts;

		/**
		 * Construct an if-else statement from a condition, true branch and
		 * optional false branch.
		 *
		 * @param condition
		 *            May not be null.
		 * @param trueBranch
		 *            A list of zero or more statements to be executed when the
		 *            condition holds; may not be null.
		 *
		 * @param alts
		 * 			A mapping from expression to the list of statements to execute
		 * 			if the expression holds. May be empty, may not be null.
		 *
		 * @param falseBranch
		 *            A list of zero of more statements to be executed when the
		 *            condition does not hold; may not be null.
		 * @param attributes
		 */
		public IfElse(Expr condition, List<Stmt> trueBranch,
				Map<Expr, List<Stmt>> alts,
				List<Stmt> falseBranch, Attribute... attributes) {
			super(attributes);
			this.condition = condition;
			this.trueBranch = new ArrayList<Stmt>(trueBranch);
			this.alts = alts;
			this.falseBranch = new ArrayList<Stmt>(falseBranch);
		}

		/**
		 * Construct an if-else statement from a condition, true branch and
		 * optional false branch.
		 *
		 * @param condition
		 *            May not be null.
		 * @param trueBranch
		 *            A list of zero or more statements to be executed when the
		 *            condition holds; may not be null.
		 *
		 * @param alts
		 * 			A mapping from expression to the list of statements to execute
		 * 			if the expression holds. May be empty, may not be null.
		 *
		 * @param falseBranch
		 *            A list of zero of more statements to be executed when the
		 *            condition does not hold; may not be null.
		 * @param attributes
		 */
		public IfElse(Expr condition, List<Stmt> trueBranch,
				Map<Expr, List<Stmt>>alts,
				List<Stmt> falseBranch, Collection<Attribute> attributes) {
			super(attributes);
			this.condition = condition;
			this.trueBranch = new ArrayList<Stmt>(trueBranch);
			this.alts = alts;
			this.falseBranch = new ArrayList<Stmt>(falseBranch);
		}

		/**
		 * Get the if-condition.
		 *
		 * @return May not be null.
		 */
		public Expr getCondition() {
			return condition;
		}

		/**
		 * Get the true branch, which consists of zero or more statements.
		 *
		 * @return May not be null.
		 */
		public List<Stmt> getTrueBranch() {
			return trueBranch;
		}

		/**
		 * Get the set of else-if expressions to check
		 *
		 * @return May not be null
		 */
		public Set<Expr> getAltExpressions() {
			return new HashSet<Expr>(alts.keySet());
		}

		/**
		 * Get the list of statements associated with a given else-if
		 * expression, or null if not an expression associated with this else
		 */
		public List<Stmt> getAltBranch(Expr e) {
			List<Stmt> tmp = alts.get(e);

			if (e == null)
				return null;

			return new ArrayList<Stmt>(tmp);
		}
		/**
		 * Get the false branch, which consists of zero or more statements.
		 *
		 * @return May not be null.
		 */
		public List<Stmt> getFalseBranch() {
			return falseBranch;
		}
	}

	/**
	 * Represents a print statement which writes expressions (as strings) to the
	 * console. The following illustrates:
	 *
	 * <pre>
	 * void f(int x):
	 *   print x+1
	 * </pre>
	 *
	 * Observe that the computed value for the expression to be printed is the
	 * implicitly coerced into a string.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Print extends SyntacticElement.Impl implements
			Atom {

		private final Expr expr;

		/**
		 * Construct a print statement from a given expression.
		 *
		 * @param expr
		 *            May not be null.
		 * @param attributes
		 */
		public Print(Expr expr, Attribute... attributes) {
			super(attributes);
			this.expr = expr;
		}

		/**
		 * Construct a print statement from a given expression.
		 *
		 * @param expr
		 *            May not be null.
		 * @param attributes
		 */
		public Print(Expr expr, Collection<Attribute> attributes) {
			super(attributes);
			this.expr = expr;
		}

		public String toString() {
			return "print " + getExpr();
		}

		/**
		 * Get the expression whose value is to be printed.
		 *
		 * @return Guaranteed to be non-null.
		 */
		public Expr getExpr() {
			return expr;
		}
	}

	/**
	 * Represents a variable declaration which is made up from a type, a
	 * variable name and an (optional) initialiser expression. If an initialiser
	 * is given, then this will be evaluated and assigned to the variable when
	 * the declaration is executed. Some example declarations:
	 *
	 * <pre>
	 * int x
	 * int y = 1
	 * int z = x + y
	 * </pre>
	 *
	 * Observe that, unlike C and Java, declarations that declare multiple
	 * variables (separated by commas) are not permitted.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class VariableDeclaration extends SyntacticElement.Impl implements
			Atom {
		private final Type type;
		private final String name;
		private final Expr expr;

		/**
		 * Construct a variable declaration from a given type, variable name and
		 * optional initialiser expression.
		 *
		 * @param type
		 *            Type of variable being declared.
		 * @param name
		 *            Name of varaible being declared.
		 * @param expr
		 *            Optional initialiser expression, which may be null.
		 * @param attributes
		 */
		public VariableDeclaration(Type type, String name, Expr expr,
				Attribute... attributes) {
			super(attributes);
			this.type = type;
			this.name = name;
			this.expr = expr;
		}

		/**
		 * Construct a variable declaration from a given type, variable name and
		 * optional initialiser expression.
		 *
		 * @param type
		 *            Type of variable being declared.
		 * @param name
		 *            Name of varaible being declared.
		 * @param expr
		 *            Optional initialiser expression, which may be null.
		 * @param attributes
		 */
		public VariableDeclaration(Type type, String name, Expr expr,
				Collection<Attribute> attributes) {
			super(attributes);
			this.type = type;
			this.name = name;
			this.expr = expr;
		}

		public String toString() {
			String r = getType() + " " + getName();
			if (getExpr() != null) {
				r = r + " = " + getExpr();
			}
			return r;
		}

		/**
		 * Get the type of the variable being declared.
		 *
		 * @return Guaranteed to be non-null.
		 */
		public Type getType() {
			return type;
		}

		/**
		 * Get the name of the variable being declared.
		 *
		 * @return Guaranteed to be non-null.
		 */
		public String getName() {
			return name;
		}

		/**
		 * Get the initialiser expression of the variable being declared (if
		 * present).
		 *
		 * @return May be null.
		 */
		public Expr getExpr() {
			return expr;
		}
	}

	/**
	 * Represents a switch statement of the form:
	 *
	 * switch (expr):
	 * 		case <literal>:
	 * 			stmt
	 * 			next  <- Denotes explicit fallthrough
	 *
	 * 		default:
	 * 			stmt
	 *
	 */
	public static final class Switch extends SyntacticElement.Impl implements
		Stmt {

		private Expr expr;
		private List<Stmt.SwitchStmt> cases;



		/**
		 * Constructs a switch statement out of a given expression and list of switch statements
		 */
		public Switch(Expr e, List<Stmt.SwitchStmt> c, Attribute...attributes) {
			super(attributes);
			expr = e;
			cases = new ArrayList<Stmt.SwitchStmt>(c);
		}

		public Expr getExpr() {
			return expr;
		}

		public List<Stmt.SwitchStmt> cases() {
			return new ArrayList<Stmt.SwitchStmt>(cases);
		}

		public String toString() {
			StringBuilder sb = new StringBuilder("switch " + expr.toString() + ":\n");
			for (Stmt.SwitchStmt c : cases) {
				sb.append(c.toString() + "\n");
			}
			return sb.toString();
		}
	}

	public static interface SwitchStmt extends Stmt {}

	public static final class Case extends SyntacticElement.Impl implements
	SwitchStmt {
		private Expr constant;  //Must be either Expr.Constant or Expr.ListConstructor
		private List<Stmt> stmts;

		public Case(Expr c, List<Stmt> s, Attribute...attributes) {
			super(attributes);
			constant = c;
			stmts = new ArrayList<Stmt>(s);
		}

		public Expr getConstant() {
			return constant;
		}

		public List<Stmt> getStmts() {
			return new ArrayList<Stmt>(stmts);
		}

		public String toString() {
			StringBuilder sb = new StringBuilder("case " + constant.toString() + ":\n");
			for (Stmt s : stmts)
				sb.append("\t" + s.toString() + "\n");
			return sb.toString();
		}
	}

	public static final class Default extends SyntacticElement.Impl implements
	SwitchStmt {
		private List<Stmt> stmts;

		public Default(List<Stmt> s, Attribute...attributes) {
			super(attributes);
			stmts = s ;
		}

		public List<Stmt> getStmts() {
			return new ArrayList<Stmt>(stmts);
		}

		public String toString() {
			StringBuilder sb = new StringBuilder("default:\n");
			for (Stmt s : stmts)
				sb.append("\t" + s.toString() + "\n");
			return sb.toString();
		}
	}
	public static final class Next extends SyntacticElement.Impl implements
	Atom {

		public Next(Attribute...attributes) {
			super(attributes);
		}

		public String toString() {
			return "next";
		}
	}
	public static final class BoundCalc {
		private Expr outer;
		private Expr inner;
		private int lowX = -1;
		private int highX = -1;
		private int lowY = -1;
		private int highY = -1;

		public BoundCalc(Stmt.ParFor loop) {
			this.outer = loop.srcX;
			this.inner = loop.srcY;
		}

		public int getLowX() {
			return lowX;
		}
		public int getHighX() {
			return highX;
		}
		public int getLowY() {
			return lowY;
		}
		public int getHighY() {
			return highY;
		}
		public void setLowX(int lowX) {
			this.lowX = lowX;
		}
		public void setHighX(int highX) {
			this.highX = highX;
		}
		public void setLowY(int lowY) {
			this.lowY = lowY;
		}
		public void setHighY(int highY) {
			this.highY = highY;
		}
		public Expr getOuter() {
			return outer;
		}
		public Expr getInner() {
			return inner;
		}
	}
}
