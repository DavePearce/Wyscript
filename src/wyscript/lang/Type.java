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

import wyscript.util.Attribute;
import wyscript.util.SyntacticElement;

/**
 * <p>
 * Represents a type as denoted in a source file (a.k.a a <i>syntactic
 * type</i>). As such types come directly from source code, they may be
 * incorrect in some fashion. For example, the type <code>{void f}</code> could
 * be written by a programmer, but is invalid type and should (eventually)
 * result in a syntax error.
 * </p>
 *
 * <p>
 * Types are not necessarily represented in their minimal form. For example, the
 * programmer may write <code>int|int</code>, which is a valid type that is
 * equivalent to <code>int</code>. Thus, further processing on types is
 * necessary if they are to be represeted in a minimal form.
 * </p>
 *
 * @author David J. Pearce
 *
 */
public interface Type extends SyntacticElement {

	/**
	 * Represents the special <code>void</code> type which can only be used in
	 * special circumstance (e.g. for a function return).
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Void extends SyntacticElement.Impl implements
			Type {

		public Void(Attribute... attributes) {
			super(attributes);
		}

		public boolean equals(Object o) {
			return o instanceof Type.Void;
		}

		public String toString() {
			return "void";
		}
	}

	/**
	 * Represents the special <code>null</code> type which can be thought of as
	 * describing a set of size one that contains the value <code>null</code>.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Null extends SyntacticElement.Impl implements
			Type {

		public Null(Attribute... attributes) {
			super(attributes);
		}

		public boolean equals(Object o) {
			return o instanceof Type.Null;
		}

		public String toString() {
			return "null";
		}
	}

	/**
	 * Represents the <code>bool</code> type which contains the values
	 * <code>true</code> and <code>false</code>.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Bool extends SyntacticElement.Impl implements
			Type {

		public Bool(Attribute... attributes) {
			super(attributes);
		}

		public boolean equals(Object o) {
			return o instanceof Type.Bool;
		}

		public String toString() {
			return "bool";
		}

	}

	/**
	 * Represents the <code>int</code> type which describes the set of all
	 * integers described in 32bit twos compliment form. For example, this is
	 * identical to a Java <code>int</code>.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Int extends SyntacticElement.Impl implements Type {

		public Int(Attribute... attributes) {
			super(attributes);
		}

		public boolean equals(Object o) {
			return o instanceof Type.Int;
		}

		public String toString() {
			return "int";
		}
	}

	/**
	 * Represents the <code>real</code> type which describess the set of all
	 * 64bit IEEE754 floating point numbers. For example, this is identical to a
	 * Java <code>double</code>.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Real extends SyntacticElement.Impl implements
			Type {

		public Real(Attribute... attributes) {
			super(attributes);
		}

		public boolean equals(Object o) {
			return o instanceof Type.Real;
		}

		public String toString() {
			return "real";
		}
	}

	/**
	 * Represents the <code>char</code> type which describes the set of all 7bit
	 * ASCII characters. Observe that this is stricly less than that described
	 * by Java's <code>char</code> type, which represents the set of UTF16
	 * values.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Char extends SyntacticElement.Impl implements
			Type {

		public Char(Attribute... attributes) {
			super(attributes);
		}

		public boolean equals(Object o) {
			return o instanceof Type.Char;
		}

		public String toString() {
			return "char";
		}
	}

	/**
	 * Represents the <code>string</code> type which describes any sequence of
	 * <code>char</code> values.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Strung extends SyntacticElement.Impl implements
			Type {
		public Strung(Attribute... attributes) {
			super(attributes);
		}

		public boolean equals(Object o) {
			return o instanceof Type.Strung;
		}

		public String toString() {
			return "string";
		}
	}

	/**
	 * Represents a named type which has yet to be expanded in the given
	 * context.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Named extends SyntacticElement.Impl implements
			Type {

		private final String name;

		public Named(String name, Attribute... attributes) {
			super(attributes);
			this.name = name;
		}

		public String toString() {
			return getName();
		}

		/**
		 * Get the name used by this type.
		 *
		 * @return
		 */
		public String getName() {
			return name;
		}

		public boolean equals(Object o) {
			if (o == null)
				return false;
			if (!(o instanceof Type.Named))
				return false;
			return name == ((Type.Named)o).name;
		}
	}

	/**
	 * Represents the type <code>[T]</code> which describes any sequence of zero
	 * or more values of type <code>T</code>.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class List extends SyntacticElement.Impl implements
			Type {

		private final Type element;

		public List(Type element, Attribute... attributes) {
			super(attributes);
			this.element = element;
		}

		/**
		 * Get the element type of this list.
		 *
		 * @return
		 */
		public Type getElement() {
			return element;
		}

		public boolean equals(Object o) {
			if (o == null)
				return false;
			if (!(o instanceof Type.List))
				return false;

			return element.equals(((Type.List)o).element);
		}

		public String toString() {
			return "[" + element.toString() + "]";
		}
	}

	/**
	 * Represents a record type, such as <code>{int x, int y}</code>, which
	 * consists of one or more (named) field types. Observe that records exhibit
	 * <i>depth</i> subtyping, but not <i>width</i> subtyping.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Record extends SyntacticElement.Impl implements
			Type {

		private final HashMap<String, Type> fields;

		public Record(Map<String, Type> fields, Attribute... attributes) {
			super(attributes);
			if (fields.size() == 0) {
				throw new IllegalArgumentException(
						"Cannot create type tuple with no fields");
			}
			this.fields = new HashMap<String, Type>(fields);
		}

		/**
		 * Get the fields which make up this record type.
		 *
		 * @return
		 */
		public Map<String, Type> getFields() {
			return fields;
		}

		public boolean equals(Object o) {
			if (o == null)
				return false;
			if (!(o instanceof Type.Record))
				return false;

			Type.Record r = (Type.Record)o;
			return fields.equals(r.getFields());
		}

		public String toString() {
			StringBuilder sb = new StringBuilder("{");
			boolean first = true;
			ArrayList<String> names = new ArrayList<String>(fields.keySet());
			Collections.sort(names);
			for (String s : names) {
				if (!first)
					sb.append(", ");
				first = false;
				sb.append(fields.get(s).toString());
				sb.append(" " + s);
			}
			sb.append("}");
			return sb.toString();
		}
	}

	/**
	 * Represents a union type, such as <code>T1|T2</code>, which describes the
	 * set union of two (or more) types. For example, the type
	 * <code>bool|null</code> describes the set <code>{true,false,null}</code>.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static final class Union extends SyntacticElement.Impl implements
			Type {

		private final ArrayList<Type> bounds;

		public Union(Collection<Type> bounds, Attribute... attributes) {
			super(attributes);

			if (bounds.size() < 2) {
				new IllegalArgumentException(
						"Cannot construct a type union with fewer than two bounds");
			}
			this.bounds = new ArrayList<Type>(bounds);
		}

		/**
		 * Get the individual types which are being unioned together.
		 *
		 * @return
		 */
		public java.util.List<Type> getBounds() {
			return bounds;
		}

		public boolean equals(Object o) {
			if (o == null)
				return false;
			if (!(o instanceof Type.Union))
				return false;

			return new HashSet<Type>(bounds).equals
					(new HashSet<Type>(((Type.Union)o).bounds));
		}

		public String toString() {
			StringBuilder sb = new StringBuilder();
			boolean first = true;
			for (Type t : bounds) {
				if (!first)
					sb.append(" | ");
				first = false;
				sb.append(t.toString());
			}
			return sb.toString();
		}
	}
}
