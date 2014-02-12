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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import wyscript.error.ParserErrorData;
import wyscript.error.ParserExprErrorData;
import wyscript.error.ParserErrorData.ErrorType;
import wyscript.io.Lexer.Token;
import wyscript.util.Attribute;
import wyscript.util.SyntacticElement;

public class WyscriptFile {

	public final String filename;

	public final ArrayList<Decl> declarations;

	public WyscriptFile(String filename, List<Decl> decls) {
		this.filename = filename;
		this.declarations = new ArrayList<Decl>(decls);
	}

	public boolean hasName(String name) {
		for (Decl d : declarations) {
			if (d instanceof ConstDecl) {
				ConstDecl cd = (ConstDecl) d;
				if (cd.name().equals(name)) {
					return true;
				}
			} else if (d instanceof TypeDecl) {
				TypeDecl cd = (TypeDecl) d;
				if (cd.name().equals(name)) {
					return true;
				}
			} else if (d instanceof FunDecl) {
				FunDecl fd = (FunDecl) d;
				if (fd.name().equals(name)) {
					return true;
				}
			} else if (d instanceof IncludeDecl) {
				IncludeDecl id = (IncludeDecl) d;
				if (id.name().equals(name)) {
					return true;
				}
			}
		}
		return false;
	}

	public ConstDecl constant(String name) {
		for (Decl d : declarations) {
			if (d instanceof ConstDecl) {
				ConstDecl cd = (ConstDecl) d;
				if (cd.name().equals(name)) {
					return cd;
				}
			}
		}
		return null;
	}

	public TypeDecl type(String name) {
		for (Decl d : declarations) {
			if (d instanceof TypeDecl) {
				TypeDecl cd = (TypeDecl) d;
				if (cd.name().equals(name)) {
					return cd;
				}
			}
		}
		return null;
	}

	public IncludeDecl include(String name) {
		for (Decl d : declarations) {
			if (d instanceof IncludeDecl) {
				IncludeDecl id = (IncludeDecl) d;
				if (id.name().equals(name)) {
					return id;
				}
			}
		}
		return null;
	}

	public List<FunDecl> functions(String name) {
		ArrayList<FunDecl> matches = new ArrayList<FunDecl>();
		for (Decl d : declarations) {
			if (d instanceof FunDecl) {
				FunDecl cd = (FunDecl) d;
				if (cd.name().equals(name)) {
					matches.add(cd);
				}
			}
		}
		return matches;
	}

	/**
	 * Returns the set of errors resulting from name clashes within this file
	 */
	public Set<ParserErrorData> checkClashes() {
		Set<ParserErrorData> errors = new HashSet<ParserErrorData>();
		Set<Decl> decs = new HashSet<Decl>();

 outer: for (Decl d : declarations) {
	 	if (d instanceof IncludeDecl)
	 		continue;

	 		for (Decl other : decs) {
				if (other.name().equals(d.name())) {
					if (other.getClass().equals(d.getClass())) {
						if (other instanceof FunDecl) {
							//Match found, add to errors
							Token.Kind kind;
							if (d instanceof TypeDecl) {
								kind = Token.Kind.Type;
							} else if (d instanceof ConstDecl)
								kind = Token.Kind.Constant;
							else
								kind = Token.Kind.Void;

							Attribute.Source source = d.attribute(Attribute.Source.class);
							Token t = new Token(null, d.toString(), 0);

							errors.add(new ParserExprErrorData(filename, new Expr.Constant(d.name()),
									t, kind, source.start, source.end, ErrorType.NAME_CLASH));

							continue outer;
						}
					}
				}
			}
			decs.add(d);
		}
		return errors;
	}

	public interface Decl extends SyntacticElement {

		public String name();
	}

	public static class ConstDecl extends SyntacticElement.Impl implements Decl {

		public final Expr constant;
		public final String name;

		public ConstDecl(Expr constant, String name, Attribute... attributes) {
			super(attributes);
			this.constant = constant;
			this.name = name;
		}

		public String name() {
			return name;
		}

		public String toString() {
			return "constant " + name + " is " + constant;
		}
	}

	public static class TypeDecl extends SyntacticElement.Impl implements Decl {

		public final Type type;
		public final String name;

		public TypeDecl(Type type, String name, Attribute... attributes) {
			super(attributes);
			this.type = type;
			this.name = name;
		}

		public String name() {
			return name;
		}

		public String toString() {
			return "type " + name + " is " + type;
		}
	}

	public static class IncludeDecl extends SyntacticElement.Impl implements
			Decl {

		public final String filepath;

		public IncludeDecl(String fp, Attribute... attributes) {
			super(attributes);
			filepath = fp;

		}

		public String name() {
			return filepath;
		}

		public String toString() {
			return "include " + filepath;
		}
	}

	public final static class FunDecl extends SyntacticElement.Impl implements
			Decl {

		public final String name;
		public final Type ret;
		public final boolean Native;

		public final ArrayList<Parameter> parameters;
		public final ArrayList<Stmt> statements;

		/**
		 * Construct an object representing a Whiley function.
		 *
		 * @param name
		 *            - The name of the function.
		 * @param ret
		 *            - The return type of this method
		 * @param parameters
		 *            - The list of parameter names and their types for this
		 *            method
		 * @param statements
		 *            - The Statements making up the function body.
		 */
		public FunDecl(String name, Type ret, List<Parameter> parameters,
				boolean Native, List<Stmt> statements, Attribute... attributes) {
			super(attributes);
			this.name = name;
			this.ret = ret;
			this.Native = Native;
			this.parameters = new ArrayList<Parameter>(parameters);
			this.statements = new ArrayList<Stmt>(statements);
		}

		public String name() {
			return name;
		}

		public String toString() {
			String params = "(";
			boolean first = true;
			for (Parameter p : parameters) {
				if (!first)
					params += ", ";
				first = false;
				params += p;
			}
			params += ")";
			return ret + " " + name + params + ":";
		}
	}

	public static final class Parameter extends SyntacticElement.Impl implements
			Decl {

		public final Type type;
		public final String name;

		public Parameter(Type type, String name, Attribute... attributes) {
			super(attributes);
			this.type = type;
			this.name = name;
		}

		public String name() {
			return name;
		}

		public boolean equals(Object other) {
			if (other == null)
				return false;
			if (other.getClass() != this.getClass())
				return false;
			Parameter p = (Parameter) other;
			return (name.equals(p.name) && type.equals(p.type));
		}

		public String toString() {
			return type + " " + name;
		}
	}
}
