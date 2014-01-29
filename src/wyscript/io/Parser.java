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

package wyscript.io;

import java.io.File;
import java.io.IOException;
import java.util.*;

import wyscript.error.ParserErrorData;
import wyscript.error.ParserErrorData.ErrorType;
import wyscript.error.ParserExprErrorData;
import static wyscript.error.ParserErrorData.ErrorType.*;
import static wyscript.error.ParserErrorHandler.*;
import wyscript.io.Lexer.*;
import static wyscript.io.Lexer.Token.Kind.*;
import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.lang.Type.Reference;
import wyscript.lang.WyscriptFile;
import wyscript.lang.WyscriptFile.*;
import wyscript.util.Attribute;
import wyscript.util.Pair;
import wyscript.util.SyntaxError;

/**
 * Responsible for parsing a sequence of tokens into an Abstract Syntax Tree.
 *
 * @author David J. Pearce
 *
 */
public class Parser {

	private String filename;
	private ArrayList<Token> tokens;
	private HashSet<String> userDefinedTypes;
	private HashSet<String> includeNames;
	private HashMap<String, Expr> constants;

	private int index;

	public Parser(String filename, List<Token> tokens) {
		this.filename = filename;
		this.tokens = new ArrayList<Token>(tokens);
		this.userDefinedTypes = new HashSet<String>();
		constants = new HashMap<String, Expr>();

		includeNames = new HashSet<String>();
		includeNames.add(filename);
	}

	/**
	 * Called when a new parser is created as part of parsing an include declaration.
	 * Used to ensure no issues with circular includes arise.
	 */
	private Parser(String filename, List<Token> tokens, Set<String> includes) {
		this(filename, tokens);
		includeNames.addAll(includes);
	}

	/**
	 * Read a <code>WyscriptFile</code> from the token stream. If the stream is
	 * invalid in some way (e.g. contains a syntax error, etc) then a
	 * <code>SyntaxError</code> is thrown.
	 *
	 * @return
	 */
	public WyscriptFile read() {
		ArrayList<Decl> decls = new ArrayList<Decl>();
		ArrayList<ParserErrorData> errors = new ArrayList<ParserErrorData>();
		skipWhiteSpace();

		while (index < tokens.size()) {
			Token t = tokens.get(index);
			Decl d;
			switch (t.kind) {

			case Type:
				d = parseTypeDeclaration(errors);
				if (d != null)
					decls.add(d);
				break;

			case Constant:
				d = parseConstantDeclaration(errors);
				if (d != null) {
					decls.add(d);
					constants.put(d.name(), ((ConstDecl)d).constant);
				}
				break;

			case Include:
				d = parseIncludeDeclaration(errors);
				if (d != null && !d.name().equals("")) {

					//First, get the absolute filepath of the included file
					String newInclude = null;
					int index = filename.lastIndexOf(File.separator);
					if (index == -1)
						newInclude = d.name();
					else newInclude = filename.substring(0, index+1) + d.name();

					//Next, check whether or not the file has already been included
					if (includeNames.contains(newInclude)) {
						skipWhiteSpace();
						continue;
					}
					includeNames.add(newInclude);

					//Finally, lex and parse the new file, and add its declarations
					WyscriptFile ast;

					try {
						File file = new File(newInclude);
						Lexer lexer = new Lexer(file.getPath());
						Parser parser = new Parser(file.getPath(), lexer.scan(), includeNames);
						ast = parser.read();
					}
					catch (IOException e) {
						Attribute.Source source = d.attribute(Attribute.Source.class);
						errors.add(new ParserExprErrorData(filename, new Expr.Constant(d.name()),
								null, null, source.start, source.end, ErrorType.BAD_INCLUDE));
						skipWhiteSpace();
						continue;
					}

					decls.addAll(ast.declarations);
				}

			default:
				d = parseFunctionDeclaration(errors);
				if (d != null)
					decls.add(d);
			}
			skipWhiteSpace();
		}

		WyscriptFile wf = new WyscriptFile(filename, decls);
		errors.addAll(wf.checkClashes());

		//Handle any errors generated during parsing, and stop compilation here
		if (!errors.isEmpty())
			handle(errors);

		return wf;
	}

	private IncludeDecl parseIncludeDeclaration(List<ParserErrorData> errors) {
		int start = index;
		boolean valid = true;

		Token.Kind follow = StringValue;
		if (match(errors, Include, follow) == null)
			valid = false;

		skipWhiteSpace();
		follow = NewLine;

		Set<Token.Kind> followSet = new HashSet<Token.Kind>();
		followSet.add(follow);

		Token string = match(errors, StringValue, followSet);
		if (string == null)
			valid = false;

		if (!matchEndLine(errors))
			valid = false;

		return (valid) ? new IncludeDecl(string.text.substring(1, string.text.length()-1), sourceAttr(start, index-1))
					   : null;
	}

	private FunDecl parseFunctionDeclaration(List<ParserErrorData> errors) {
		int start = index;
		boolean valid = true;
		boolean isNative = false;

		if (tryAndMatch(Native, true) != null) {
			isNative = true;
		}
		Token.Kind follow = Token.Kind.Identifier;
		Set<Token.Kind> followSet = new HashSet<Token.Kind>();
		followSet.add(follow);
		if (match(errors, Function, follow) == null) {
			valid = false;
			if (! followSet.contains(tokens.get(index))) {
				index++;
				return null;
			}

		}

		follow = Token.Kind.LeftBrace;
		followSet = new HashSet<Token.Kind>();
		followSet.add(follow);

		skipWhiteSpace();
		follow = LeftBrace;

		Token name = match(errors, Identifier, follow);
		if (name == null)
			valid = false;

		follow = RightBrace;
		if(match(errors, LeftBrace, follow) == null)
			valid = false;

		followSet.add(RightBrace);
		followSet.remove(Identifier);

		// Now build up the parameter types
		List<Parameter> paramTypes = new ArrayList<Parameter>();
		boolean firstTime = true;
		while (eventuallyMatch(errors, RightBrace) == null) {

			if (!firstTime) {
				if (match(errors, Comma, followSet) == null) {
					valid = false;
					break;
				}
			}

			firstTime = false;
			int pstart = index;

			Type t = parseType(errors, true, followSet);
			if (t == null) {
				valid = false;
				if (tokens.get(index).kind == RightBrace)
					break;
			}

			Token n = match(errors, Identifier, followSet);
			if (n == null) {
				valid = false;
				break;
			}


			if (valid)
				paramTypes.add(new Parameter(t, n.text, sourceAttr(pstart,
					index - 1)));
		}

		follow = Colon;
		followSet.add(Colon);
		followSet.remove(RightBrace);

		Token n = match(errors, DoubleArrow, followSet);
		if (n == null)
			valid = false;

		Type ret = parseType(errors, true, followSet);

		//If parsing a native function, stop here
		if (isNative) {
			if (!matchEndLine(errors))
				valid = false;

			return (valid) ? new FunDecl(name.text, ret, paramTypes, true, new ArrayList<Stmt>(), sourceAttr(start, index-1)) :
							 new FunDecl("", new Type.Void(), new ArrayList<Parameter>(), true, new ArrayList<Stmt>());
		}

		follow = NewLine;
		if (match(errors, Colon, follow) == null)
			valid = false;

		if (!matchEndLine(errors))
			valid = false;

		List<Stmt> stmts = parseBlock(ROOT_INDENT, errors, new HashSet<Token.Kind>());

		if (stmts == null) {
			valid = false;
		}

		return (valid) ? new FunDecl(name.text, ret, paramTypes, false, stmts, sourceAttr(start,
				index - 1))
					   : new FunDecl("", new Type.Void(), new ArrayList<Parameter>(), false, new ArrayList<Stmt>());
	}

	private Decl parseTypeDeclaration(List<ParserErrorData> errors) {

		int start = index;
		boolean valid = true;

		Token.Kind follow = Identifier;
		if(match(errors, Type, follow) == null)
			valid = false;

		follow = Is;
		Token id = match(errors, Identifier, follow);
		if (id == null)
			valid = false;

		follow = NewLine;
		if(match(errors, Is, follow) == null) {
			//Must have matched a newline, need to increment index then return null
			index++;
			return null;
		}

		Set<Token.Kind> followSet = new HashSet<Token.Kind>();
		followSet.add(follow);

		Type t = parseType(errors, true, followSet);
		if (t == null)
			valid = false;

		int end = index;
		matchEndLine(errors);
		userDefinedTypes.add(id.text);
		return (valid) ? new TypeDecl(t, id.text, sourceAttr(start, end - 1))
					   : new TypeDecl(new Type.Void(), "");
	}

	private Decl parseConstantDeclaration(List<ParserErrorData> errors) {
		int start = index;
		boolean valid = true;

		Token.Kind follow = Identifier;
		match(errors, Constant, follow);

		follow = Is;
		Token id = match(errors, Identifier, follow);
		if (id == null)
			valid = false;

		follow = NewLine;
		if (match(errors, Is, follow) == null) {
			index++;
			return new ConstDecl(null, "");
		}

		Set<Token.Kind> followSet = new HashSet<Token.Kind>();
		followSet.add(follow);

		Expr e = parseTupleExpression(errors, true, followSet);
		if (e == null)
			valid = false;

		int end = index;
		matchEndLine(errors);

		return (valid) ? new ConstDecl(e, id.text, sourceAttr(start, end - 1))
					   : new ConstDecl(null, "");
	}

	/**
	 * Parse a block of zero or more statements which share the same indentation
	 * level. Their indentation level must be strictly greater than that of
	 * their parent, otherwise the end of block is signaled. The <i>indentation
	 * level</i> for the block is set by the first statement encountered
	 * (assuming their is one). An error occurs if a subsequent statement is
	 * reached with an indentation level <i>greater</i> than the block's
	 * indentation level.
	 *
	 * @param parentIndent
	 *            The indentation level of the parent, for which all statements
	 *            in this block must have a greater indent. May not be
	 *            <code>null</code>.
	 * @return
	 */
	private List<Stmt> parseBlock(Indent parentIndent, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {
		// First, determine the initial indentation of this block based on the
		// first statement (or null if there is no statement).
		Indent indent = getIndent();

		// Second, check that this is indeed the initial indentation for this
		// block (i.e. that it is strictly greater than parent indent).
		if (indent == null || indent.lessThanEq(parentIndent)) {
			// Initial indent either doesn't exist or is not strictly greater
			// than parent indent and,therefore, signals an empty block.
			//
			return Collections.EMPTY_LIST;
		} else {
			// Initial indent is valid, so we proceed parsing statements with
			// the appropriate level of indent.
			//
			ArrayList<Stmt> stmts = new ArrayList<Stmt>();
			Indent nextIndent;
			while ((nextIndent = getIndent()) != null
					&& indent.lessThanEq(nextIndent)) {
				// At this point, nextIndent contains the indent of the current
				// statement. However, this still may not be equivalent to this
				// block's indentation level.

				// First, check the indentation matches that for this block.
				if (!indent.equivalent(nextIndent)) {
					// No, it's not equivalent so signal an error.
					errors.add(new ParserErrorData(filename, nextIndent, Indent, BAD_INDENT));
				}

				// Second, parse the actual statement at this point!
				Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
				followSet.add(NewLine);
				Stmt tmp = (parseStatement(indent, errors, followSet));
				if (tmp != null) stmts.add(tmp);
				else {
					Token token = tokens.get(index);
					if (token.kind == NewLine)
						index++;
					else {
						//Was a member of the follow set, so we need to return null to indicate
						return null;
					}
				}
			}

			return stmts;
		}
	}

	/**
	 * Determine the indentation as given by the Indent token at this point (if
	 * any). If none, then <code>null</code> is returned. If a newline is encountered, skip
	 * over until an indent or a non-whitespace token is found
	 *
	 * @return
	 */
	private Indent getIndent() {
		int pos = index;
		while (pos < tokens.size() && isWhiteSpace(tokens.get(index))) {
			Token token = tokens.get(pos);
			if(token.kind == Indent) {
				return new Indent(token.text,token.start);
			}
			pos ++;
		}
		return null;
	}

	/**
	 * Parse a given statement. There are essentially two forms of statement:
	 * <code>simple</code> and <code>compound</code>. Simple statements (e.g.
	 * assignment, <code>print</code>, etc) are always occupy a single line and
	 * are terminated by a <code>NewLine</code> token. Compound statements (e.g.
	 * <code>if</code>, <code>while</code>, etc) themselves contain blocks of
	 * statements and are not (generally) terminated by a <code>NewLine</code>.
	 *
	 * @param indent
	 *            The indent level for the current statement. This is needed in
	 *            order to constraint the indent level for any sub-blocks (e.g.
	 *            for <code>while</code> or <code>if</code> statements).
	 *
	 * @return
	 */
	private Stmt parseStatement(Indent indent, List<ParserErrorData> errors, Set<Token.Kind> follow) {
		checkNotEof(errors, Statement);
		Token token = tokens.get(index++);

		switch(token.kind) {
		case Return:
			return parseReturnStatement(index-1, errors, follow);
		case Print:
			return parsePrintStatement(index-1, errors, follow);
		case If:
			return parseIfStatement(index-1,indent, errors, follow);
		case While:
			return parseWhile(index-1,indent, errors, follow);
		case For:
			return parseFor(index-1,indent, errors, follow);

		case Next:
			return parseNext(index-1, errors, follow);

		case Switch:
			return parseSwitch(index-1, indent, errors, follow);
		case Identifier:
			if (tryAndMatch(Token.Kind.LeftBrace, false) != null) {
				return parseInvokeStatement(token, errors, follow);
			}
		}

		index = index - 1; // backtrack
		if (isStartOfType(index)) {
			return parseVariableDeclaration(errors, follow);
		} else {
			return parseAssign(errors, follow);
		}
	}

	/**
	 * Determine whether or not a given position marks the beginning of a type
	 * declaration or not. This is important to help determine whether or not
	 * this is the beginning of a variable declaration.
	 *
	 * @param index
	 *            Position in the token stream to begin looking from.
	 * @return
	 */
	private boolean isStartOfType(int index) {
		if (index >= tokens.size()) {
			return false;
		}

		Token token = tokens.get(index);
		switch(token.kind) {
		case Void:
		case Null:
		case Bool:
		case Int:
		case Real:
		case Char:
		case String:
			return true;
		case Identifier:
			return userDefinedTypes.contains(token.text);
		case LeftCurly:
		case LeftSquare:
		case Ampersand:
		case LogicalAnd:
			return isStartOfType(index + 1);

		//More complex, have to check for the comma to ensure isn't a cast
		case LeftBrace:

			if (!(isStartOfType(index + 1)))
				return false;

			int next = index + 1;
			int scopeCount = 0;
			while (next < tokens.size()) {
				if (tokens.get(next).kind == Comma && scopeCount == 0)
					return true;
				if (tokens.get(next).kind == RightBrace && scopeCount == 0)
					return false;
				if (tokens.get(next).kind == RightBrace)
					scopeCount--;
				if (tokens.get(next).kind == LeftBrace)
					scopeCount++;

				next++;
			}
			return false;
		}

		return false;
	}

	/**
	 * A simple statement only of use within a switch case body, where it signals
	 * a fall-through to the next case body.
	 *
	 */
	private Stmt.Next parseNext(int start, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		matchEndLine(errors);
		return new Stmt.Next(sourceAttr(start, start));
	}

	/**
	 * Parse an invoke statement, which has the form:
	 *
	 * <pre>
	 * Identifier '(' ( Expression )* ')' NewLine
	 * </p>
	 *
	 * Observe that this when this function is called, we're assuming that the identifier and opening brace has already been matched.
	 *
	 * @return
	 */
	private Expr.Invoke parseInvokeStatement(Token name, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		int start = index - 2;
		boolean valid = true;

		// An invoke statement begins with the name of the function to be
		// invoked, followed by zero or more comma-separated arguments enclosed
		// in braces.
		boolean firstTime = true;
		ArrayList<Expr> args = new ArrayList<Expr>();
		Token.Kind follow = RightBrace;

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(follow);

		while (eventuallyMatch(errors, RightBrace) == null) {
			if (!firstTime) {
				if (match(errors,Token.Kind.Comma, followSet) == null) {
					//Need to check where control flow should go
					if (tokens.get(index).kind == RightBrace) {
						index++;
						valid = false;
						break;
					}
					else return null;
				}
			} else {
				firstTime = false;
			}
			Expr e = parseLogicalExpression(errors, true, followSet);
			if (e == null) {
				valid = false;
				if (tokens.get(index).kind == RightBrace)
					valid = false;
				else return null;
			}
			if (valid) args.add(e);

		}
		// Finally, a new line indicates the end-of-statement
		int end = index;
		matchEndLine(errors);
		// Done
		return (valid) ? new Expr.Invoke(name.text, args, sourceAttr(start, end - 1))
					   : new Expr.Invoke("", new ArrayList<Expr>());
	}

	/**
	 * Parse a variable declaration statement, which has the form:
	 *
	 * <pre>
	 * Type Identifier ['=' Expression] NewLine
	 * </pre>
	 *
	 * The optional <code>Expression</code> assignment is referred to as an
	 * <i>initialiser</i>.
	 *
	 * @return
	 */
	private Stmt.VariableDeclaration parseVariableDeclaration(List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		int start = index;
		boolean valid = true;

		// Every variable declaration consists of a declared type and variable
		// name.
		Token.Kind follow = Identifier;
		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(follow);

		Type type = parseType(errors, true, followSet);
		if (type == null) {
			valid = false;
			if (!(tokens.get(index).kind == Identifier))
				return null; //Have synchronized to parent method
		}

		if (!parentFollow.contains(Identifier))
			followSet.remove(Identifier);

		followSet.add(Equals);
		followSet.add(NewLine);

		Token id = match(errors, Identifier, followSet);
		if (id == null) {
			valid = false;
			switch (tokens.get(index).kind) {

			case Equals:
				break;

			case NewLine:
				index++;
				return new Stmt.VariableDeclaration(null, "", null);

			default:
				return null;
			}
		}

		// A variable declaration may optionally be assigned an initialiser
		// expression.
		Expr initialiser = null;
		if (tryAndMatch(Token.Kind.Equals, false) != null) {
			if (!parentFollow.contains(Equals))
				followSet.remove(Equals);
			initialiser = parseTupleExpression(errors, false, followSet);
			if (initialiser == null) {
				valid = false;
				if (tokens.get(index).kind != NewLine)
					return null;
			}
		}
		// Finally, a new line indicates the end-of-statement
		int end = index;
		matchEndLine(errors);
		// Done.
		return (valid) ? new Stmt.VariableDeclaration(type, id.text, initialiser,
				sourceAttr(start, end - 1))
					   : new Stmt.VariableDeclaration(null, "", null);
	}

	/**
	 * Parse a return statement, which has the form:
	 *
	 * <pre>
	 * "return" [Expression] NewLine
	 * </pre>
	 *
	 * The optional expression is referred to as the <i>return value</i>.
	 * Observe that, when this function is called, we're assuming that "return"
	 * has already been matched.
	 *
	 * @return
	 */
	private Stmt.Return parseReturnStatement(int start, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		Expr e = null;
		boolean valid = true;

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(NewLine);

		// A return statement may optionally have a return expression.
		int next = skipLineSpace(index);
		if (next < tokens.size() && tokens.get(next).kind != NewLine) {
			e = parseTupleExpression(errors, false, followSet);
			if (e == null) {
				valid = false;
				if (tokens.get(index).kind != NewLine)
					return null;
			}
		}
		// Finally, a new line indicates the end-of-statement
		int end = index;
		matchEndLine(errors);
		// Done.
		return (valid) ? new Stmt.Return(e, sourceAttr(start, end - 1))
					   : new Stmt.Return(null);
	}

	/**
	 * Parse a print statement, which has the form:
	 *
	 * <pre>
	 * "print" Expression
	 * </pre>
	 *
	 * Observe that, when this function is called, we're assuming that "print"
	 * has already been matched.
	 *
	 * @return
	 */
	private Stmt.Print parsePrintStatement(int start, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		// A print statement begins with the keyword "print", followed by the
		// expression whos value will be printed.
		boolean valid = true;
		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(NewLine);

		Expr e = parseTupleExpression(errors, false, followSet);
		if (e == null) {
			valid = false;
			if (tokens.get(index).kind != NewLine)
				return null;
		}

		// Finally, a new line indicates the end-of-statement
		int end = index;
		matchEndLine(errors);
		// Done
		return (valid) ? new Stmt.Print(e, sourceAttr(start, end - 1))
					   : new Stmt.Print(null);
	}

	/**
	 * Parse an if statement, which is has the form:
	 *
	 * <pre>
	 * if Expression ':' NewLine Block ["else" ':' NewLine Block]
	 * </pre>
	 *
	 * As usual, the <code>else</block> is optional.
	 *
	 * @param indent
	 * @return
	 */
	private Stmt parseIfStatement(int start, Indent indent, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {
		// An if statement begins with the keyword "if", followed by an
		// expression representing the condition.
		boolean valid = true;
		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		Map<Expr, List<Stmt>> alts = new HashMap<Expr, List<Stmt>>(); //The map of else-if alternatives
		followSet.add(Colon);

		Expr c = parseLogicalExpression(errors, true, followSet);
		if (c == null) {
			valid = false;
			if (tokens.get(index).kind != Colon)
				return null;
		}

		// The a colon to signal the start of a block.
		if (!parentFollow.contains(Colon))
			followSet.remove(Colon);
		followSet.add(NewLine);

		if (match(errors, Colon, followSet) == null) {
			if (tokens.get(index).kind != NewLine)
				return null;
		}

		matchEndLine(errors);
		if (!parentFollow.contains(NewLine))
			followSet.remove(NewLine);
		int end = index;

		followSet.add(Else);

		// First, parse the true branch, which is required
		List<Stmt> tblk = parseBlock(indent, errors, followSet);
		if (tblk == null) {
			valid = false;
			if (tokens.get(index).kind != Else)
				return null;
		}

		// Second, attempt to parse the false branch, which is optional.
		List<Stmt> fblk = Collections.emptyList();
		if (tryAndMatch(Else, true) != null) {

			if (!parentFollow.contains(Else))
				followSet.remove(Else);
			followSet.add(Colon);

			//Match any number of else-if statements
			while (tryAndMatch(If, true) != null) {
				Expr e = parseLogicalExpression(errors, true, followSet);
				if (e == null) {
					valid = false;
					if (tokens.get(index).kind != Colon)
						return null;
				}
				if (!parentFollow.contains(Colon))
					followSet.remove(Colon);
				followSet.add(NewLine);

				if (match(errors, Colon, followSet) == null) {
					valid = false;
					if (tokens.get(index).kind != NewLine)
						return null;
				}
				matchEndLine(errors);
				if (!parentFollow.contains(NewLine))
					followSet.remove(NewLine);
				followSet.add(Else);

				List<Stmt> blk = parseBlock(indent, errors, followSet);
				if (blk == null) {
					valid = false;
					if (tokens.get(index).kind != Else)
						return null;
				}
				//Add to the map of else-if alternatives
				if (valid)
					alts.put(e, blk);

				if (! parentFollow.contains(Else))
					followSet.remove(Else);
				if (tryAndMatch(Else, true) == null) {
					return (valid) ? new Stmt.IfElse(c, tblk, alts, fblk, sourceAttr(start, end - 1))
					   			   : new Stmt.IfElse(null, new ArrayList<Stmt>(),
					   					   new HashMap<Expr, List<Stmt>>(), new ArrayList<Stmt>());
				}

			}
			if (match(errors, Colon, followSet) == null) {
				if (tokens.get(index).kind != NewLine);
				return null;
			}
			matchEndLine(errors);
			fblk = parseBlock(indent, errors, parentFollow);
			if (fblk == null)
				return null;
		}
		// Done!
		return (valid) ? new Stmt.IfElse(c, tblk, alts, fblk, sourceAttr(start, end - 1))
					   : new Stmt.IfElse(null, new ArrayList<Stmt>(), new HashMap<Expr, List<Stmt>>(), new ArrayList<Stmt>());
	}

	/**
	 * Parse a while statement, which has the form:
	 * <pre>
	 * "while" Expression ':' NewLine Block
	 * </pre>
	 * @param indent
	 * @return
	 */
	private Stmt parseWhile(int start, Indent indent, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		boolean valid = true;
		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(Colon);

		Expr condition = parseLogicalExpression(errors, true, followSet);
		if (condition == null) {
			valid = false;
			if (tokens.get(index).kind != Colon)
				return null;
		}

		if (!parentFollow.contains(Colon))
			followSet.remove(Colon);
		followSet.add(NewLine);

		if (match(errors, Colon, followSet) == null) {
			if (tokens.get(index).kind != NewLine)
				return null;
		}

		int end = index;
		matchEndLine(errors);
		List<Stmt> blk = parseBlock(indent, errors, parentFollow);
		if (blk == null)
			return null;

		return (valid) ? new Stmt.While(condition, blk, sourceAttr(start, end - 1))
					   : new Stmt.While(null, new ArrayList<Stmt>());
	}

	private Stmt parseSwitch(int start, Indent indent, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		boolean valid = true;
		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(Colon);

		Expr expr = parseTupleExpression(errors, true, followSet);
		if (expr == null) {
			valid = false;
			if (tokens.get(index).kind != Colon)
				return null;
		}

		if (!parentFollow.contains(Colon))
			followSet.remove(Colon);
		followSet.add(NewLine);

		if (match(errors, Colon, followSet) == null) {
			if (tokens.get(index).kind != NewLine)
				return null;
		}

		int end = index;
		matchEndLine(errors);
		if (!parentFollow.contains(NewLine))
			followSet.remove(NewLine);

		List<Stmt.SwitchStmt> cases = parseCases(indent, errors, parentFollow);
		if (cases == null) {
			return null;
		}

		return (valid) ? new Stmt.Switch(expr, cases, sourceAttr(start, end - 1))
					   : new Stmt.Switch(null, new ArrayList<Stmt.SwitchStmt>());
	}

	/**
	 * Parses a set of cases for a switch statement - similar in structure to parsing
	 * a block, with the additional caveat that only case statements are allowed, and
	 * no two case statements may have the same label
	 */
	private List<Stmt.SwitchStmt> parseCases(Indent parentIndent, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		Indent indent = getIndent();
		int start = index;
		index = skipLineSpace(index);
		//Check for the trivial case where the switch body is empty
		if (indent == null || indent.lessThanEq(parentIndent) || index >= tokens.size() ||
				(tokens.get(index).kind != Case && tokens.get(index).kind != Default)) {
			return Collections.EMPTY_LIST;
		}

		else {
			index = start;
			ArrayList<Stmt.SwitchStmt> stmts = new ArrayList<Stmt.SwitchStmt>();
			Indent nextIndent;
			Set<String> usedLiterals = new HashSet<String>();
			boolean def = true;

			while ((nextIndent = getIndent()) != null
					&& indent.lessThanEq(nextIndent)) {

				//Check the indent
				if (!indent.equivalent(nextIndent)) {
					errors.add(new ParserErrorData(filename, nextIndent, Indent, BAD_INDENT));
				}

				//Matching a case statement
				if (tryAndMatch(Case, true) != null) {

					Stmt.Case tmp = (parseCase(indent, usedLiterals, errors, parentFollow));
					if (tmp != null) {
						stmts.add(tmp);
						if (tmp.getConstant() != null)
							usedLiterals.add(tmp.getConstant().toString());
					}
					else {
						//Was a member of the follow set, so we need to return null to indicate
						return null;
					}
				}
				//Matching a default statement
				else if (tryAndMatch(Default, true) != null) {
					if (def) {
						def = false;
					}
					else {
						errors.add(new ParserErrorData(filename, tokens.get(index-1), Token.Kind.Case, SWITCH_MULTIPLE_DEFAULT));
					}
					Stmt.Default d = (parseDefault(indent, errors, parentFollow));
					if (d == null)
						return null;
					else stmts.add(d);
				}
				else {
					index = skipLineSpace(index);
					errors.add(new ParserErrorData(filename, tokens.get(index), Token.Kind.Case, BAD_SWITCH_CASE));
					synchronize(NewLine, new HashSet<Token.Kind>(), errors);
				}
			}
			return stmts;
		}
	}

	private Stmt.Case parseCase(Indent indent, Set<String> usedLiterals, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		int start = index -1;

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(Colon);
		boolean valid = true;

		Expr e = parseLogicalExpression(errors, true, followSet);

		if (e == null) {
			valid = false;
			if (tokens.get(index).kind != Colon)
				return null;
		}

		if (valid && !(e instanceof Expr.Constant || e instanceof Expr.ListConstructor)) {

			//Constant is ok
			if (e instanceof Expr.Variable &&
					constants.get(((Expr.Variable) e).getName()) != null);
			else {
				valid = false;
				Attribute.Source loc = e.attribute(Attribute.Source.class);
				errors.add(new ParserExprErrorData(filename, e, null, Constant, loc.start, loc.end, BAD_SWITCH_CONST));
			}
		}

		if (valid && usedLiterals.contains(e.toString())) {
			valid = false;
			Attribute.Source loc = e.attribute(Attribute.Source.class);
			errors.add(new ParserExprErrorData(filename, e, null, Constant, loc.start, loc.end, DUPLICATE_SWITCH_CONST));
		}

		if (!parentFollow.contains(Colon))
			followSet.remove(Colon);
		followSet.add(NewLine);

		match(errors, Colon, followSet);
		matchEndLine(errors);

		List<Stmt> stmts = parseBlock(indent, errors, parentFollow);
		if (stmts == null)
			return null;
		int end = index;

		return (valid) ? new Stmt.Case(e, stmts, sourceAttr(start, end-1))
					   : new Stmt.Case(null, new ArrayList<Stmt>());
	}

	private Stmt.Default parseDefault(Indent indent, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		int start = index-1;
		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(NewLine);
		match(errors, Colon, followSet);
		matchEndLine(errors);

		List<Stmt> stmts = parseBlock(indent, errors, parentFollow);
		if (stmts == null)
			return null;
		int end = index;

		return new Stmt.Default(stmts, sourceAttr(start, end-1));
	}


	private Stmt parseFor(int start, Indent indent, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		boolean valid = true;
		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(In);

		Token id = match(errors, Identifier, followSet);
		if (id == null) {
			valid = false;
			if (tokens.get(index).kind != In)
				return null;
		}

		Expr.Variable var =  (valid) ? new Expr.Variable(id.text, sourceAttr(start,
				index - 1))
									 : null;

		if (!parentFollow.contains(In))
			followSet.remove(In);
		followSet.add(Colon);

		Expr source = null;
		if(match(errors, In, followSet) == null) {
			valid = false;
			if (tokens.get(index).kind != Colon)
				return null;
		}
		else {
			source = parseLogicalExpression(errors, true, followSet);
			if (source == null) {
				valid = false;
				if (tokens.get(index).kind != Colon)
					return null;
			}
		}

		if (!parentFollow.contains(Colon))
			followSet.remove(Colon);
		followSet.add(NewLine);

		if (match(errors, Colon, followSet) == null) {
			if (tokens.get(index).kind != NewLine)
				return null;
		}

		int end = index;
		matchEndLine(errors);
		List<Stmt> blk = parseBlock(indent, errors, parentFollow);
		if (blk == null)
			return null;

		return (valid) ? new Stmt.For(var, source, blk, sourceAttr(start, end - 1))
					   : new Stmt.For(null, null, new ArrayList<Stmt>());
	}

	/**
	 * Parse an assignment statement of the form "lval = expression".
	 *
	 * @return
	 */
	private Stmt parseAssign(List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow ) {

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(Equals);
		boolean valid = true;

		// standard assignment
		int start = index;
		Expr lhs = parseTupleExpression(errors, true, followSet);
		if (lhs == null) {
			valid = false;
			if (tokens.get(index).kind != Equals)
				return null;
		}

		else if (!(lhs instanceof Expr.LVal)) {
			errors.add(new ParserExprErrorData(filename, lhs, tokens.get(start), ExprLval, lhs.attribute(Attribute.Source.class).start,
					lhs.attribute(Attribute.Source.class).end, ErrorType.BAD_EXPRESSION_TYPE));
			valid = false;
		}

		if (!parentFollow.contains(Equals))
			followSet.remove(Equals);
		followSet.add(NewLine);

		Expr rhs = null;

		if (match(errors, Equals, followSet) == null) {
			valid = false;
			if (tokens.get(index).kind != NewLine)
				return null;
		}
		else {
			rhs = parseTupleExpression(errors, false, followSet);
			if (rhs == null) {
				valid = false;
				if (tokens.get(index).kind != NewLine)
					return null;
			}
		}

		int end = index;
		matchEndLine(errors);

		return (valid) ? new Stmt.Assign((Expr.LVal) lhs, rhs, sourceAttr(start, end - 1))
					   : new Stmt.Assign(null, null);
	}

	private Expr parseTupleExpression(List<ParserErrorData> errors, boolean terminated, Set<Token.Kind> parentFollow) {

		checkNotEof(errors, Expression);

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(Comma);

		int start = index;

		Expr e = parseLogicalExpression(errors, terminated, followSet);

		if (e == null) {
			if (tokens.get(index).kind != Comma)
				return null;
		}

		int next = (terminated) ? skipWhiteSpace(index) : skipLineSpace(index);
		if (!(next < tokens.size() && tokens.get(next).kind == Comma))
			return e;

		index = next + 1;

		List<Expr> exprs = new ArrayList<Expr>();
		exprs.add(e);

		do {
			e = parseLogicalExpression(errors, terminated, followSet);
			if (e == null) {
				if (tokens.get(index).kind != Comma)
					return null;
			}
			else exprs.add(e);
		}
		while (tryAndMatch(Comma, terminated) != null);

		return new Expr.Tuple(exprs, sourceAttr(start, index-1));
	}

	private Expr parseLogicalExpression(List<ParserErrorData> errors, boolean terminated, Set<Token.Kind> parentFollow) {

		checkNotEof(errors, Expression);

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(LogicalAnd);
		followSet.add(LogicalOr);

		int start = index;
		Expr lhs = parseConditionExpression(errors, terminated, followSet);
		if (lhs == null) {
			switch (tokens.get(index).kind) {

			case LogicalAnd:
			case LogicalOr:
				break;

			default:
				return null;
			}
		}
		int next = (terminated) ? skipWhiteSpace(index) : skipLineSpace(index);

		while (next < tokens.size()) {
			Token token = tokens.get(next);
			Expr.BOp bop;
			switch (token.kind) {

			case LogicalOr:
				bop = Expr.BOp.OR;
				break;
			case LogicalAnd:
				bop = Expr.BOp.AND;
				break;
			default:
				return lhs;
			}
			index = next+1; // match the operator
			Expr term = parseConditionExpression(errors, terminated, parentFollow);
			if (term == null)
				return null;

			else {
				next = (terminated) ? skipWhiteSpace(index) : skipLineSpace(index);
				lhs = new Expr.Binary(bop, lhs, term, sourceAttr(start, index - 1));
			}
		}
		return lhs;
	}

	private Expr parseConditionExpression(List<ParserErrorData> errors,
			boolean terminated, Set<Token.Kind> parentFollow) {

		int start = index;

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(LessEquals);
		followSet.add(LeftAngle);
		followSet.add(GreaterEquals);
		followSet.add(RightAngle);
		followSet.add(EqualsEquals);
		followSet.add(NotEquals);
		followSet.add(Is);

		Expr lhs = parseAppendExpression(errors, terminated, followSet);
		if (lhs == null) {
			switch (tokens.get(index).kind) {

			case LessEquals:
			case LeftAngle:
			case GreaterEquals:
			case RightAngle:
			case EqualsEquals:
			case NotEquals:
			case Is:
				break;

			default:
				return null;
			}
		}

		int next = (terminated) ? skipWhiteSpace(index) : skipLineSpace(index);
		boolean hasMatched = false;
		while (next < tokens.size()) {
			Token token = tokens.get(next);
			Expr.BOp bop;
			switch (token.kind) {
			case LessEquals:
				bop = Expr.BOp.LTEQ;
				break;
			case LeftAngle:
				bop = Expr.BOp.LT;
				break;
			case GreaterEquals:
				bop = Expr.BOp.GTEQ;
				break;
			case RightAngle:
				bop = Expr.BOp.GT;
				break;
			case EqualsEquals:
				bop = Expr.BOp.EQ;
				break;
			case NotEquals:
				bop = Expr.BOp.NEQ;
				break;
			case Is:
				if (hasMatched)
					return lhs;
				index = next + 1; // match the operator
				Type rhs = parseType(errors, terminated, parentFollow);
				if (rhs == null)
					return null;
				else
					return new Expr.Is(lhs, rhs, sourceAttr(start, index - 1));
			default:
				return lhs;
			}

			index = next + 1; // match the operator
			Expr term = parseAppendExpression(errors, terminated, parentFollow);
			if (term == null)
				return null;

			hasMatched = true; //prevents parsing a statement like (a <= b is bool)
			next = (terminated) ? skipWhiteSpace(index) : skipLineSpace(index);
			lhs = new Expr.Binary(bop, lhs, term, sourceAttr(start, index - 1));
		}
		return lhs;
	}


	private Expr parseAppendExpression(List<ParserErrorData> errors,
			boolean terminated, Set<Token.Kind> parentFollow) {

		int start = index;

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(PlusPlus);

		Expr lhs = parseRangeExpression(errors, terminated, followSet);
		if (lhs == null) {
			if (tokens.get(index).kind != PlusPlus)
				return null;
		}

		int next = (terminated) ? skipWhiteSpace(index) : skipLineSpace(index);
		if (next < tokens.size()) {
			Token token = tokens.get(next);
			switch (token.kind) {
			case PlusPlus:
				index = next + 1; // match the operator
				Expr rhs = parseAppendExpression(errors, terminated, parentFollow);
				if (rhs == null)
					return null;

				return new Expr.Binary(Expr.BOp.APPEND, lhs, rhs, sourceAttr(start,
						index - 1));
			}
		}
		return lhs;
	}

	private Expr parseRangeExpression(List<ParserErrorData> errors,
			boolean terminated, Set<Token.Kind> parentFollow) {

		int start = index;

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(DotDot);

		Expr lhs = parseAddSubExpression(errors, terminated, followSet);
		if (lhs == null) {
			if (tokens.get(index).kind != DotDot)
				return null;
		}

		int next = (terminated) ? skipWhiteSpace(index) : skipLineSpace(index);
		if (next < tokens.size()) {
			Token token = tokens.get(next);
			switch (token.kind) {
			case DotDot:
				index = next + 1; // match the operator
				Expr rhs = parseRangeExpression(errors, terminated, parentFollow);
				if (rhs == null)
					return null;

				return new Expr.Binary(Expr.BOp.RANGE, lhs, rhs, sourceAttr(start,
						index - 1));
			}
		}
		return lhs;
	}

	private Expr parseAddSubExpression(List<ParserErrorData> errors,
			boolean terminated, Set<Token.Kind> parentFollow ) {

		int start = index;

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(Plus);
		followSet.add(Minus);

		Expr lhs = parseMulDivExpression(errors, terminated, followSet);
		if (lhs == null) {
			switch (tokens.get(index).kind) {
			case Plus:
			case Minus:
				break;

			default:
				return null;
			}
		}

		int next = (terminated) ? skipWhiteSpace(index) : skipLineSpace(index);
		while (next < tokens.size()) {
			Token token = tokens.get(next);
			Expr.BOp bop;
			switch (token.kind) {
			case Plus:
				bop = Expr.BOp.ADD;
				break;
			case Minus:
				bop = Expr.BOp.SUB;
				break;
			default:
				return lhs;
			}
			index = next + 1; // match the operator
			Expr rhs = parseMulDivExpression(errors, terminated, parentFollow);
			if (rhs == null)
				return null;

			next = (terminated) ? skipWhiteSpace(index) : skipLineSpace(index);
			lhs = new Expr.Binary(bop, lhs, rhs, sourceAttr(start, index - 1));
		}
		return lhs;
	}

	private Expr parseMulDivExpression(List<ParserErrorData> errors,
			boolean terminated, Set<Token.Kind> parentFollow) {

		int start = index;

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(Star);
		followSet.add(RightSlash);
		followSet.add(Percent);

		Expr lhs = parseIndexTerm(errors, terminated, followSet);
		if (lhs == null) {
			switch (tokens.get(index).kind) {

			case Star:
			case RightSlash:
			case Percent:
				break;

			default:
				return null;
			}
		}


		int next = (terminated) ? skipWhiteSpace(index) : skipLineSpace(index);
		while (next < tokens.size()) {
			Token token = tokens.get(next);
			Expr.BOp bop;
			switch (token.kind) {
			case RightSlash:
				bop = Expr.BOp.DIV;
				break;
			case Percent:
				bop = Expr.BOp.REM;
				break;
			case Star:
				bop = Expr.BOp.MUL;
				break;
			default:
				return lhs;
			}
			index = next + 1; // match the operator
			Expr rhs = parseIndexTerm(errors, terminated, parentFollow);
			if (rhs == null)
				return null;

			next = (terminated) ? skipWhiteSpace(index) : skipLineSpace(index);
			lhs = new Expr.Binary(bop, lhs, rhs, sourceAttr(start, index - 1));
		}
		return lhs;
	}

	private Expr parseIndexTerm(List<ParserErrorData> errors,
			boolean terminated, Set<Token.Kind> parentFollow) {

		int start = index;

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(LeftSquare);
		followSet.add(Dot);
		followSet.add(Arrow);

		Expr lhs = parseTerm(errors, terminated, followSet);
		if (lhs == null) {
			switch(tokens.get(index).kind) {

			case LeftSquare:
			case Dot:
			case Arrow:
				break;

			default:
				return null;
			}
		}

		Token token;

		while ((token = tryAndMatch(LeftSquare, terminated)) != null
				|| (token = tryAndMatch(Dot, terminated)) != null
				|| (token = tryAndMatch(Arrow, terminated)) != null) {
			start = index;

			if (!parentFollow.contains(LeftSquare))
				followSet.remove(LeftSquare);
			if (!parentFollow.contains(Dot))
				followSet.remove(Dot);
			if (!parentFollow.contains(Arrow))
				followSet.remove(Arrow);

			if (token.kind == LeftSquare) {

				followSet.add(RightSquare);

				Expr rhs = parseAddSubExpression(errors, true, followSet);
				if (rhs == null)
					if (tokens.get(index).kind != RightSquare)
						return null;

				if (match(errors, RightSquare, parentFollow) == null)
					return null;

				lhs = new Expr.IndexOf(lhs, rhs, sourceAttr(start, index - 1));
			}
			else {
				Token id = match(errors, Identifier, parentFollow);
				if (id == null)
					return null;

				lhs = (token.kind == Arrow) ? new Expr.RecordAccess(new Expr.Deref(lhs), id.text, sourceAttr(start,index - 1))
											: new Expr.RecordAccess(lhs, id.text, sourceAttr(start,index - 1));
			}
		}
		return lhs;
	}

	private Expr parseTerm(List<ParserErrorData> errors,
			boolean terminated, Set<Token.Kind> parentFollow) {

		checkNotEof(errors, Expression);

		int start = index;
		Token token = tokens.get(index++);

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);

		switch(token.kind) {
		case LeftBrace:
			followSet.add(RightBrace);

			if (isStartOfType(index)) {
				// indicates a cast

				Type t = parseType(errors, true, followSet);
				if (t == null)
					if (tokens.get(index).kind != RightBrace)
						return null;

				if (match(errors, RightBrace, parentFollow) == null)
					return null;
				Expr e = parseTerm(errors, terminated, parentFollow);

				if (e == null)
					return null;

				return new Expr.Cast(t, e, sourceAttr(start, index - 1));
			}
			else {
				Expr e = parseTupleExpression(errors, true, followSet);
				if (e == null)
					if (tokens.get(index).kind != RightBrace)
						return null;

				if (match(errors, RightBrace, parentFollow) == null)
					return null;
				if (e != null)
					e.attributes().add(new Attribute.Parentheses());
				return e;
			}

		case Identifier:
			if (tryAndMatch(LeftBrace, terminated) != null) {
				return parseInvokeExpr(start,token, errors, parentFollow);
			} else {
				return new Expr.Variable(token.text, sourceAttr(start,
						index - 1));
			}
		case Null:
			return new Expr.Constant(null, sourceAttr(start, index - 1));
		case True:
			return new Expr.Constant(true, sourceAttr(start, index - 1));
		case False:
			return new Expr.Constant(false, sourceAttr(start, index - 1));
		case CharValue:
			return new Expr.Constant(parseCharacter(token.text), sourceAttr(
					start, index - 1));
		case IntValue:
			return new Expr.Constant(Integer.parseInt(token.text), sourceAttr(
					start, index - 1));
		case RealValue:
			return new Expr.Constant(Double.parseDouble(token.text), sourceAttr(
					start, index - 1));
		case StringValue:
			String str = parseString(token.text);
			return new Expr.Constant(new StringBuffer(str), sourceAttr(start,
					index - 1));
		case Minus:
			return parseNegation(start, errors, terminated, parentFollow);
		case VerticalBar:
			return parseLengthOf(start, errors, parentFollow);
		case LeftSquare:
			return parseListVal(start, errors, parentFollow);
		case LeftCurly:
			return parseRecordVal(start, errors, parentFollow);
		case Shreak:
			Expr tmp = parseTerm(errors, terminated, parentFollow);
			if (tmp == null)
				return null;
			return new Expr.Unary(Expr.UOp.NOT, tmp, sourceAttr(start,
					index - 1));

		case New:
			Expr e = parseTupleExpression(errors, terminated, parentFollow);
			if (e == null)
				return null;
			return new Expr.New(e, sourceAttr(start, index-1));

		case Star:
			Expr ex = parseTerm(errors, terminated, parentFollow);
			if (ex == null)
				return null;
			return new Expr.Deref(ex, sourceAttr(start, index-1));
		}

		//Couldn't parse, may have hit garbage values. Skip until we match something in the follow set
		errors.add(new ParserErrorData(filename, token, null, MISSING_EXPRESSION));
		synchronize(null, parentFollow, errors);
		return null;
	}

	private Expr parseListVal(int start, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		ArrayList<Expr> exprs = new ArrayList<Expr>();
		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(RightSquare);

		boolean firstTime = true;
		while (eventuallyMatch(errors, RightSquare) == null) {
			if (!firstTime) {
				if (match(errors, Comma, followSet) == null) {
					if (tokens.get(index).kind == RightSquare) {
						index++;
						break;
					}
					else return null;
				}
			}
			firstTime = false;
			Expr e = parseLogicalExpression(errors, true, followSet);
			if (e == null) {
				if (tokens.get(index).kind != RightSquare)
					return null;
			}
			else exprs.add(e);
		}
		return new Expr.ListConstructor(exprs, sourceAttr(start, index - 1));
	}

	private Expr parseRecordVal(int start, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		HashSet<String> keys = new HashSet<String>();
		ArrayList<Pair<String, Expr>> exprs = new ArrayList<Pair<String, Expr>>();
		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(RightCurly);

		boolean firstTime = true;
		outer: while (eventuallyMatch(errors, RightCurly) == null) {
			if (!firstTime) {
				if (match(errors, Comma, followSet) == null) {
					if (tokens.get(index).kind == RightCurly) {
						index++;
						break;
					}
					else return null;
				}
			}

			firstTime = false;
			boolean valid = true;
			followSet.add(Colon);
			checkNotEof(errors, Identifier);
			Token n = match(errors, Identifier, followSet);
			if (n == null) {
				valid = false;
				switch(tokens.get(index).kind) {

				case RightCurly:
					index++;
					break outer;

				case Colon:
					break;

				default:
					return null;
				}
			}

			if (valid && keys.contains(n.text)) {
				valid = false;
				errors.add(new ParserErrorData(filename, n, null, DUPLICATE_TOKEN));
			}

			if (!parentFollow.contains(Colon))
				followSet.remove(Colon);

			if (match(errors, Colon, followSet) == null) {
				if (tokens.get(index).kind == RightCurly) {
					index++;
					break;
				}
				else return null;
			}

			Expr e = parseLogicalExpression(errors, true, followSet);
			if (e == null) {
				valid = false;
				if (tokens.get(index).kind != RightCurly)
					return null;
			}

			if (valid) {
				exprs.add(new Pair<String, Expr>(n.text, e));
				keys.add(n.text);
			}
		}
		return new Expr.RecordConstructor(exprs, sourceAttr(start, index - 1));
	}

	private Expr parseLengthOf(int start, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow ) {

		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(VerticalBar);

		Expr e = parseIndexTerm(errors, true, followSet);
		if (e == null) {
			if (tokens.get(index).kind != VerticalBar)
				return null;
		}

		if (match(errors, VerticalBar, parentFollow) == null)
			return null;

		return new Expr.Unary(Expr.UOp.LENGTHOF, e,
				sourceAttr(start, index - 1));
	}

	private Expr parseNegation(int start, List<ParserErrorData> errors,
			boolean terminated, Set<Token.Kind> parentFollow) {

		Expr e = parseIndexTerm(errors, terminated, parentFollow);
		if (e == null)
			return null;

		if (e instanceof Expr.Constant) {
			Expr.Constant c = (Expr.Constant) e;
			if (c.getValue() instanceof Integer) {
				int bi = (Integer) c.getValue();
				return new Expr.Constant(-bi, sourceAttr(start, index));
			} else if (c.getValue() instanceof Double) {
				double br = (Double) c.getValue();
				return new Expr.Constant(-br, sourceAttr(start, index));
			}
		}
		return new Expr.Unary(Expr.UOp.NEG, e, sourceAttr(start, index));
	}

	private Expr.Invoke parseInvokeExpr(int start, Token name, List<ParserErrorData> errors,
			Set<Token.Kind> parentFollow) {

		boolean firstTime = true;
		ArrayList<Expr> args = new ArrayList<Expr>();
		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(RightBrace);

		while (eventuallyMatch(errors, RightBrace) == null) {
			if (!firstTime) {
				if (match(errors, Comma, followSet) == null) {
					if (tokens.get(index).kind == RightBrace) {
						index++;
						break;
					}
					else return null;
				}
			} else {
				firstTime = false;
			}
			boolean valid = true;
			Expr e = parseLogicalExpression(errors, true, followSet);
			if (e == null) {
				valid = false;
				if (tokens.get(index).kind != RightBrace)
					return null;
			}
			if (valid)
				args.add(e);
		}
		return new Expr.Invoke(name.text, args, sourceAttr(start, index - 1));
	}

	private Type parseType(List<ParserErrorData> errors,
			boolean terminated, Set<Token.Kind> parentFollow) {

		int start = index;
		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		followSet.add(VerticalBar);

		Type t = parseBaseType(errors, terminated, followSet);
		if (t == null) {
			if (tokens.get(index).kind != VerticalBar)
				return null;
		}


		// Now, attempt to look for union types
		if (tryAndMatch(VerticalBar, terminated) != null) {
			// this is a union type
			ArrayList<Type> types = new ArrayList<Type>();
			types.add(t);

			do {
				Type tmp = parseBaseType(errors, terminated, followSet);
				if (tmp == null) {
					if (tokens.get(index).kind != VerticalBar)
						return null;
				}
				else
					types.add(tmp);
			}
			while (tryAndMatch(VerticalBar, terminated) != null);

			return new Type.Union(types, sourceAttr(start, index - 1));
		} else {
			return t;
		}
	}

	private Type parseBaseType(List<ParserErrorData> errors,
			boolean terminated, Set<Token.Kind> parentFollow) {

		checkNotEof(errors, Type2);
		int start = index;
		Token token = tokens.get(index++);
		Set<Token.Kind> followSet = new HashSet<Token.Kind>(parentFollow);
		Type t;

		switch (token.kind) {
		case Null:
			return new Type.Null(sourceAttr(start, index - 1));
		case Void:
			return new Type.Void(sourceAttr(start, index - 1));
		case Bool:
			return new Type.Bool(sourceAttr(start, index - 1));
		case Char:
			return new Type.Char(sourceAttr(start, index - 1));
		case Int:
			return new Type.Int(sourceAttr(start, index - 1));
		case Real:
			return new Type.Real(sourceAttr(start, index - 1));
		case String:
			return new Type.Strung(sourceAttr(start, index - 1));
		case LeftCurly:
			HashMap<String, Type> types = new HashMap<String, Type>();

			boolean firstTime = true;

			followSet.add(RightCurly);

			while (eventuallyMatch(errors, RightCurly) == null) {
				if (!firstTime) {
					if (match(errors, Comma, followSet) == null) {
						if (tokens.get(index).kind == RightCurly) {
							index++;
							break;
						}
						else return null;
					}
				}
				firstTime = false;
				boolean valid = true;

				checkNotEof(errors, Type2);
				Type tmp = parseType(errors, true, followSet);
				if (tmp == null) {
					valid = false;
					if (tokens.get(index).kind == RightCurly) {
						index++;
						break;
					}
					else return null;
				}

				Token n = match(errors, Identifier, followSet);
				if (n == null) {
					valid = false;
					if (tokens.get(index).kind == RightCurly) {
						index++;
						break;
					}
					else return null;
				}


				if (valid && types.containsKey(n.text)) {
					errors.add(new ParserErrorData(filename, n, null, DUPLICATE_TOKEN));
					valid = false;
				}
				if (valid)
					types.put(n.text, tmp);
			}

			return new Type.Record(types, sourceAttr(start, index - 1));

		case LeftSquare:

			followSet.add(RightSquare);

			t = parseType(errors, true, followSet);
			if (t == null) {
				if (tokens.get(index).kind != RightSquare)
					return null;
			}

			if (match(errors, RightSquare, parentFollow) == null)
				return null;

			return new Type.List(t, sourceAttr(start, index - 1));

		case LeftBrace:

			followSet.add(RightBrace);
			List<Type> typeList = new ArrayList<Type>();

			boolean first = true;
			while (eventuallyMatch(errors, RightBrace) == null) {
				if (!first) {
					if (match(errors, Comma, followSet) == null) {
						if (tokens.get(index).kind == RightBrace) {
							index++;
							break;
						}
						else return null;
					}
				}
				first = false;
				boolean valid = true;

				checkNotEof(errors, Type2);
				Type tmp = parseType(errors, true, followSet);
				if (tmp == null) {
					valid = false;
					if (tokens.get(index).kind == RightBrace) {
						index++;
						break;
					}
					else return null;
				}
				if (valid)
					typeList.add(tmp);
			}
			return new Type.Tuple(typeList, sourceAttr(start, index-1));

		case Identifier:
			return new Type.Named(token.text, sourceAttr(start, index - 1));

		case Ampersand:
		case LogicalAnd:
			t = parseType(errors, terminated, parentFollow);
			if (t == null)
				return null;

			return (token.kind == Ampersand) ? new Type.Reference(t, sourceAttr(start, index -1))
											 : new Type.Reference(new Type.Reference(t), sourceAttr(start, index-1));

		default:
			errors.add(new ParserErrorData(filename, token, null, INVALID_TYPE));
			//Unable to find type, may be on garbled data, synchronize with parent methods
			synchronize(null, parentFollow, errors);
			return null;
		}
	}

	/**
	 * Match a given token kind, whilst moving passed any whitespace encountered
	 * inbetween. In the case that meet the end of the stream, or we don't match
	 * the expected token, then an error is thrown.
	 *
	 * @param kind
	 * @return
	 */
	private Token match(List<ParserErrorData> errors, Token.Kind kind,
			Set<Token.Kind> followSet) {

		int start = (index > 0) ? tokens.get(index-1).end()+1 : tokens.get(index).end()+1;
		int startIndex = index;
		checkNotEof(errors, kind);
		Token token = tokens.get(index++);

		if (token.kind != kind) {
			errors.add(new ParserErrorData(filename, token, kind, start, start, MISSING_TOKEN));
			index--;

			//Have to deal with the case where checkNotEof eats the \n we were looking for
			if (followSet.contains(NewLine))
				index = startIndex;
			if (synchronize(kind, followSet, errors))
				return tokens.get(index++);
			return null;
		}
		return token;
	}

	/**
	 * Identical to above, but used to simplify cases where only
	 * one token is in the follow set
	 */
	private Token match(List<ParserErrorData> errors, Token.Kind kind,
			Token.Kind follow) {
		Set<Token.Kind> followSet = new HashSet<Token.Kind>();
		followSet.add(follow);
		return match(errors, kind, followSet);
	}

	/**
	 * Utility method for the parser's error recovery system - skips tokens until either
	 * the expected token is found, or a member of the follow set is found.
	 * Ends parsing if EOF is encountered.
	 *
	 * @param expected 	- The expected token
	 * @param follow	- The set of following tokens
	 *
	 * @return true if the expected token was found, false otherwise
	 */
	private boolean synchronize(Token.Kind expected, Set<Token.Kind> follow,
			List<ParserErrorData> errors) {

		while (    tokens.size() > index
				&& !follow.contains(tokens.get(index).kind)
				&& tokens.get(index).kind != expected) {
			index++;
		}
		checkNotEof(errors, expected);
		if (tokens.get(index).kind == expected)
			return true;
		return false;
	}

	/**
	 * Attempt to match a given kind of token with the view that it must
	 * *eventually* be matched. This differs from <code>tryAndMatch()</code>
	 * because it calls <code>checkNotEof()</code>. Thus, it is guaranteed to
	 * skip any whitespace encountered in between. This is safe because we know
	 * there is a terminating token still to come.
	 *
	 * @param kind
	 * @return
	 */
	private Token eventuallyMatch(List<ParserErrorData> errors, Token.Kind kind) {
		checkNotEof(errors, kind);
		Token token = tokens.get(index);
		if (token.kind != kind) {
			return null;
		} else {
			index = index + 1;
			return token;
		}
	}

	/**
	 * Attempt to match a given token, whilst ignoring any whitespace in
	 * between. Note that, in the case it fails to match, then the index will be
	 * unchanged. This latter point is important, otherwise we could
	 * accidentally gobble up some important indentation.
	 *
	 * @param kind
	 * @param terminated - whether or not the current expression has a terminator
	 * 					   (and so whether or not we can safely skip newlines)
	 * @return
	 */
	private Token tryAndMatch(Token.Kind kind, boolean terminated) {
		int next = (terminated) ? skipWhiteSpace(index) : skipLineSpace(index);
		if(next < tokens.size()) {
			Token t = tokens.get(next);
			if(t.kind == kind) {
				index = next + 1;
				return t;
			}
		}
		return null;
	}

	/**
	 * Match a the end of a line. This is required to signal, for example, the
	 * end of the current statement.
	 */
	private boolean matchEndLine(List<ParserErrorData> errors) {
		// First, parse all whitespace characters except for new lines
		index = skipLineSpace(index);

		// Second, check whether we've reached the end-of-file (as signaled by
		// running out of tokens), or we've encountered some token which not a
		// newline.
		if (index >= tokens.size()) {
			int pos = (index > 0) ? tokens.get(index-1).end()+1 : tokens.get(index).end()+1;
			errors.add(new ParserErrorData(filename, null, NewLine, pos, pos, MISSING_TOKEN));
			handle(errors);
			return false;
		}
		else if (tokens.get(index).kind != NewLine) {
			errors.add(new ParserErrorData(filename, tokens.get(index), NewLine, MISSING_TOKEN));
			synchronize(NewLine, new HashSet<Token.Kind>(), errors);
			index++;
			return true;
		} else {
			index = index + 1;
			return true;
		}
	}

	/**
	 * Check that the End-Of-File has not been reached. This method should be
	 * called from contexts where we are expecting something to follow.
	 */
	private void checkNotEof(List<ParserErrorData> errors, Token.Kind expected) {
		int start = (index > 0) ? tokens.get(index-1).end()+1 : tokens.get(index).end()+1;
		int indexStart = index;
		skipWhiteSpace();

		//Work around to deal with cases where we are looking for a NewLine
		if (expected == NewLine) {
			index = skipLineSpace(indexStart);
		}

		if (index >= tokens.size()) {
			errors.add(new ParserErrorData(filename, null, expected, start, start, MISSING_TOKEN));
			handle(errors);
		}
	}


	/**
	 * Skip over any whitespace characters.
	 */
	private void skipWhiteSpace() {
		index = skipWhiteSpace(index);
	}

	/**
	 * Skip over any whitespace characters, starting from a given index and
	 * returning the first index passed any whitespace encountered.
	 */
	private int skipWhiteSpace(int index) {
		while (index < tokens.size() && isWhiteSpace(tokens.get(index))) {
			index++;
		}
		return index;
	}

	/**
	 * Skip over any whitespace characters that are permitted on a given line
	 * (i.e. all except newlines), starting from a given index and returning the
	 * first index passed any whitespace encountered.
	 */
	private int skipLineSpace(int index) {
		while (index < tokens.size() && isLineSpace(tokens.get(index))) {
			index++;
		}
		return index;
	}

	/**
	 * Define what is considered to be whitespace.
	 *
	 * @param token
	 * @return
	 */
	private boolean isWhiteSpace(Token token) {
		return token.kind == Token.Kind.NewLine || isLineSpace(token);
	}

	/**
	 * Define what is considered to be linespace.
	 *
	 * @param token
	 * @return
	 */
	private boolean isLineSpace(Token token) {
		return token.kind == Token.Kind.Indent;
	}

	/**
	 * Parse a character from a string of the form 'c' or '\c'.
	 *
	 * @param input
	 * @return
	 */
	public char parseCharacter(String input) {
		int pos = 1;
		char c = input.charAt(pos++);
		if (c == '\\') {
			// escape code
			switch (input.charAt(pos++)) {
			case 't':
				c = '\t';
				break;
			case 'n':
				c = '\n';
				break;
			default:
				throw new RuntimeException("unrecognised escape character");
			}
		}
		return c;
	}

	/**
	 * Parse a string whilst interpreting all escape characters.
	 *
	 * @param v
	 * @return
	 */
	protected String parseString(String v) {
		/*
		 * Parsing a string requires several steps to be taken. First, we need
		 * to strip quotes from the ends of the string.
		 */
		v = v.substring(1, v.length() - 1);
		// Second, step through the string and replace escaped characters
		for (int i = 0; i < v.length(); i++) {
			if (v.charAt(i) == '\\') {
				if (v.length() <= i + 1) {
					throw new RuntimeException("unexpected end-of-string");
				} else {
					char replace = 0;
					int len = 2;
					switch (v.charAt(i + 1)) {
					case 'b':
						replace = '\b';
						break;
					case 't':
						replace = '\t';
						break;
					case 'n':
						replace = '\n';
						break;
					case 'f':
						replace = '\f';
						break;
					case 'r':
						replace = '\r';
						break;
					case '"':
						replace = '\"';
						break;
					case '\'':
						replace = '\'';
						break;
					case '\\':
						replace = '\\';
						break;
					case 'u':
						len = 6; // unicode escapes are six digits long,
						// including "slash u"
						String unicode = v.substring(i + 2, i + 6);
						replace = (char) Integer.parseInt(unicode, 16); // unicode
						break;
					default:
						throw new RuntimeException("unknown escape character");
					}
					v = v.substring(0, i) + replace + v.substring(i + len);
				}
			}
		}
		return v;
	}


	private Attribute.Source sourceAttr(int start, int end) {
		Token t1 = tokens.get(start);
		Token t2 = tokens.get(end);
		return new Attribute.Source(t1.start, t2.end());
	}

	private void syntaxError(String msg, Expr e) {
		Attribute.Source loc = e.attribute(Attribute.Source.class);
		throw new SyntaxError(msg, filename, loc.start, loc.end);
	}

	private void syntaxError(String msg, Token t) {
		throw new SyntaxError(msg, filename, t.start, t.start + t.text.length()
				- 1);
	}

	/**
	 * Represents a given amount of indentation. Specifically, a count of tabs
	 * and spaces. Observe that the order in which tabs / spaces occurred is not
	 * retained.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static class Indent extends Token {
		private final int countOfSpaces;
		private final int countOfTabs;

		public Indent(String text, int pos) {
			super(Token.Kind.Indent, text, pos);
			// Count the number of spaces and tabs
			int nSpaces = 0;
			int nTabs = 0;
			for (int i = 0; i != text.length(); ++i) {
				char c = text.charAt(i);
				switch (c) {
				case ' ':
					nSpaces++;
					break;
				case '\t':
					nTabs++;
					break;
				default:
					throw new IllegalArgumentException(
							"Space or tab character expected");
				}
			}
			countOfSpaces = nSpaces;
			countOfTabs = nTabs;
		}

		/**
		 * Test whether this indentation is considered "less than or equivalent"
		 * to another indentation. For example, an indentation of 2 spaces is
		 * considered less than an indentation of 3 spaces, etc.
		 *
		 * @param other
		 *            The indent to compare against.
		 * @return
		 */
		public boolean lessThanEq(Indent other) {
			return countOfSpaces <= other.countOfSpaces
					&& countOfTabs <= other.countOfTabs;
		}

		/**
		 * Test whether this indentation is considered "equivalent" to another
		 * indentation. For example, an indentation of 3 spaces followed by 1
		 * tab is considered equivalent to an indentation of 1 tab followed by 3
		 * spaces, etc.
		 *
		 * @param other
		 *            The indent to compare against.
		 * @return
		 */
		public boolean equivalent(Indent other) {
			return countOfSpaces == other.countOfSpaces
					&& countOfTabs == other.countOfTabs;
		}
	}

	/**
	 * An abstract indentation which represents the indentation of top-level
	 * declarations, such as function declarations. This is used to simplify the
	 * code for parsing indentation.
	 */
	private static final Indent ROOT_INDENT = new Indent("", 0);

	public Set<String> getUserTypes() {
		return userDefinedTypes;
	}

}
