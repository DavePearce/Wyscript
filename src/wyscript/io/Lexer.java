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

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import wyscript.error.LexerErrorData;
import static wyscript.error.LexerErrorHandler.handle;

/**
 * Responsible for turning a stream of characters into a sequence of tokens.
 *
 * @author David J. Pearce
 *
 */
public class Lexer {

	private String filename;
	private StringBuffer input;
	private int pos;

	public Lexer(String filename) throws IOException {
		this(new InputStreamReader(new FileInputStream(filename), "UTF8"));
		this.filename = filename;
	}

	public Lexer(InputStream instream) throws IOException {
		this(new InputStreamReader(instream, "UTF8"));
	}

	public Lexer(Reader reader) throws IOException {
		BufferedReader in = new BufferedReader(reader);

		StringBuffer text = new StringBuffer();
		String tmp;
		while ((tmp = in.readLine()) != null) {
			text.append(tmp);
			text.append("\n");
		}

		input = text;
	}

	/**
	 * Scan all characters from the input stream and generate a corresponding
	 * list of tokens, whilst discarding all whitespace and comments.
	 * Stores a list of all errors encountered, in order to be able to
	 * attempt error recovery
	 *
	 * @return
	 */
	public List<Token> scan() {
		ArrayList<Token> tokens = new ArrayList<Token>();
		ArrayList<LexerErrorData> errors = new ArrayList<LexerErrorData>();
		pos = 0;

		while (pos < input.length()) {
			char c = input.charAt(pos);

			if (Character.isDigit(c)) {
				tokens.add(scanNumericConstant());
			} else if (c == '"') {
				tokens.add(scanStringConstant(errors));
			} else if (c == '\'') {
				tokens.add(scanCharacterConstant(errors));
			} else if (isOperatorStart(c)) {
				tokens.add(scanOperator(errors));
			} else if (Character.isJavaIdentifierStart(c)) {
				tokens.add(scanIdentifier());
			} else if(Character.isWhitespace(c)) {
				scanWhiteSpace(tokens, errors);
			} else {
				//Skip over the offending token for now
				errors.add(new LexerErrorData(pos++, filename, c, LexerErrorData.ErrorType.INVALID_CHARACTER));
			}
		}
		if (!errors.isEmpty())
			handle(errors);

		return tokens;
	}

	/**
	 * Scan a numeric constant. That is a sequence of digits which gives either
	 * an integer constant, or a real constant (if it includes a dot).
	 *
	 * @return
	 */
	public Token scanNumericConstant() {
		int start = pos;
		while (pos < input.length() && Character.isDigit(input.charAt(pos))) {
			pos = pos + 1;
		}
		if (pos < input.length() && input.charAt(pos) == '.') {
			pos = pos + 1;
			if (pos < input.length() && input.charAt(pos) == '.') {
				// this is case for range e.g. 0..1
				pos = pos - 1;
				return new Token(Token.Kind.IntValue, input.substring(start,
						pos), start);
			}
			while (pos < input.length() && Character.isDigit(input.charAt(pos))) {
				pos = pos + 1;
			}
			return new Token(Token.Kind.RealValue, input.substring(start, pos),
					start);
		} else {
			return new Token(Token.Kind.IntValue, input.substring(start, pos),
					start);
		}
	}

	/**
	 * Scan a character constant, such as e.g. 'c'. Observe that care must be
	 * taken to properly handle escape codes. For example, '\n' is a single
	 * character constant which is made up from two characters in the input
	 * string.
	 *
	 * @return
	 */
	public Token scanCharacterConstant(List<LexerErrorData> errors) {
		int start = pos;
		pos++;
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
				errors.add(new LexerErrorData(pos-1, filename, input.charAt(pos-1), LexerErrorData.ErrorType.INVALID_ESCAPE));
				c = '\n';
			}
		}
		if (input.charAt(pos) != '\'') {
			//This simulates adding a closing quote
			pos--;
			errors.add(new LexerErrorData(pos, filename, input.charAt(pos), LexerErrorData.ErrorType.MISSING_CHAR_END));
		}
		pos = pos + 1;
		return new Token(Token.Kind.CharValue, input.substring(start, pos),
				start);
	}

	public Token scanStringConstant(List<LexerErrorData> errors) {
		int start = pos;
		pos++;
		while (pos < input.length()) {
			char c = input.charAt(pos);
			if (c == '"') {
				String v = input.substring(start, ++pos);
				return new Token(Token.Kind.StringValue, v, start);
			}
			pos = pos + 1;
		}
		errors.add(new LexerErrorData(pos-1, filename, null, LexerErrorData.ErrorType.MISSING_STRING_END));
		return new Token(Token.Kind.StringValue, input.substring(start, pos), start);
	}


	static final char[] opStarts = { ',', '(', ')', '[', ']', '{', '}', '+',
			'-', '*', '/', '%', '!', '=', '<', '>', ':', ';', '&', '|',
			'.'};

	public boolean isOperatorStart(char c) {
		for (char o : opStarts) {
			if (c == o) {
				return true;
			}
		}
		return false;
	}

	public Token scanOperator(List<LexerErrorData> errors) {
		char c = input.charAt(pos);

		switch(c) {
		case '.':
			if((pos+1) < input.length() && input.charAt(pos+1) == '.') {
				pos = pos + 2;
				return new Token(Token.Kind.DotDot,"..",pos);
			} else {
				return new Token(Token.Kind.Dot,".",pos++);
			}
		case  ',':
			return new Token(Token.Kind.Comma,",",pos++);
		case  ';':
			return new Token(Token.Kind.SemiColon,";",pos++);
		case ':':
			return new Token(Token.Kind.Colon,":",pos++);

		case '|':
			if((pos+1) < input.length() && input.charAt(pos+1) == '.') {
				pos = pos+2;
				return new Token(Token.Kind.LogicalOr, "||", pos);
			}
				return new Token(Token.Kind.VerticalBar,"|",pos++);

		case '(':
			return new Token(Token.Kind.LeftBrace,"(",pos++);
		case ')':
			return new Token(Token.Kind.RightBrace,")",pos++);
		case '[':
			return new Token(Token.Kind.LeftSquare,"[",pos++);
		case ']':
			return new Token(Token.Kind.RightSquare,"]",pos++);
		case '{':
			return new Token(Token.Kind.LeftCurly,"{",pos++);
		case '}':
			return new Token(Token.Kind.RightCurly,"}",pos++);
		case '+':
			if((pos+1) < input.length() && input.charAt(pos+1) == '+') {
				pos = pos + 2;
				return new Token(Token.Kind.PlusPlus,"++",pos);
			} else {
				return new Token(Token.Kind.Plus,"+",pos++);
			}
		case '-':
			return new Token(Token.Kind.Minus,"-",pos++);
		case '*':
			return new Token(Token.Kind.Star,"*",pos++);
		case '&':
			if (pos + 1 < input.length()
				&& input.charAt(pos + 1) == '&') {
				pos += 2;
				return new Token(Token.Kind.LogicalAnd,"&&", pos - 2);
			}
			break;
		case '/':
			return new Token(Token.Kind.RightSlash,"/",pos++);
		case '%':
			return new Token(Token.Kind.Percent,"%",pos++);
		case '!':
			if ((pos + 1) < input.length() && input.charAt(pos + 1) == '=') {
				pos += 2;
				return new Token(Token.Kind.NotEquals, "!=", pos - 2);
			} else {
				return new Token(Token.Kind.Shreak,"!",pos++);
			}
		case '=':
			if ((pos + 1) < input.length() && input.charAt(pos + 1) == '=') {
				pos += 2;
				return new Token(Token.Kind.EqualsEquals,"==",pos - 2);
			} else {
				return new Token(Token.Kind.Equals,"=",pos++);
			}
		case '<':
			if ((pos + 1) < input.length() && input.charAt(pos + 1) == '=') {
				pos += 2;
				return new Token(Token.Kind.LessEquals, "<=", pos - 2);
			} else {
				return new Token(Token.Kind.LeftAngle, "<", pos++);
			}
		case '>':
			if ((pos + 1) < input.length() && input.charAt(pos + 1) == '=') {
				pos += 2;
				return new Token(Token.Kind.GreaterEquals,">=", pos - 2);
			} else {
				return new Token(Token.Kind.RightAngle,">",pos++);
			}
		}

		//Shouldn't be possible, but will handle it anyway
		errors.add(new LexerErrorData(pos, filename, c, LexerErrorData.ErrorType.INVALID_OP));

		//Semicolon is not used for anything, so makes a good dud value
		return new Token(Token.Kind.SemiColon, ";", pos++);
	}

	public Token scanIdentifier() {
		int start = pos;
		while (pos < input.length()
				&& Character.isJavaIdentifierPart(input.charAt(pos))) {
			pos++;
		}
		String text = input.substring(start, pos);

		// now, check for keywords
		Token.Kind kind = keywords.get(text);
		if (kind == null) {
			// not a keyword, so just a regular identifier.
			kind = Token.Kind.Identifier;
		}
		return new Token(kind, text, start);
	}

	public void scanWhiteSpace(List<Token> tokens, List<LexerErrorData> errors) {
		while (pos < input.length()
				&& Character.isWhitespace(input.charAt(pos))) {
			if (input.charAt(pos) == ' ' || input.charAt(pos) == '\t') {
				tokens.add(scanIndent());
			} else if (input.charAt(pos) == '\n') {
				tokens.add(new Token(Token.Kind.NewLine, input.substring(pos,
						pos + 1), pos));
				pos = pos + 1;
			} else if (input.charAt(pos) == '\r' && (pos + 1) < input.length()
					&& input.charAt(pos + 1) == '\n') {
				tokens.add(new Token(Token.Kind.NewLine, input.substring(pos,
						pos + 2), pos));
				pos = pos + 2;
			} else {
				errors.add(new LexerErrorData(pos, filename, input.charAt(pos),
						LexerErrorData.ErrorType.INVALID_WHITESPACE));

				//Just skip over the bad whitespace
				pos++;
			}
		}
	}

	/**
	 * Scan one or more spaces or tab characters, combining them to form an
	 * "indent".
	 *
	 * @return
	 */
	public Token scanIndent() {
		int start = pos;
		while (pos < input.length()
				&& (input.charAt(pos) == ' ' || input.charAt(pos) == '\t')) {
			pos++;
		}
		return new Token(Token.Kind.Indent, input.substring(start, pos), start);
	}

	/**
	 * Skip over any whitespace at the current index position in the input
	 * string.
	 *
	 * @param tokens
	 */
	public void skipWhitespace(List<Token> tokens) {
		while (pos < input.length()
				&& (input.charAt(pos) == '\n' || input.charAt(pos) == '\t')) {
			pos++;
		}
	}

	/**
	 * A map from identifier strings to the corresponding token kind.
	 */
	public static final HashMap<String, Token.Kind> keywords = new HashMap<String, Token.Kind>() {
		{
			put("void", Token.Kind.Void);
			put("null", Token.Kind.Null);
			put("bool", Token.Kind.Bool);
			put("int", Token.Kind.Int);
			put("real", Token.Kind.Real);
			put("char", Token.Kind.Char);
			put("string", Token.Kind.String);
			put("true", Token.Kind.True);
			put("false", Token.Kind.False);
			put("if", Token.Kind.If);
			put("else", Token.Kind.Else);
			put("switch", Token.Kind.Switch);
			put("while", Token.Kind.While);
			put("for", Token.Kind.For);
			put("print", Token.Kind.Print);
			put("return", Token.Kind.Return);
			put("constant", Token.Kind.Constant);
			put("type", Token.Kind.Type);
			put("is", Token.Kind.Is);
			put("in", Token.Kind.In);
		}
	};

	/**
	 * The base class for all tokens.
	 *
	 * @author David J. Pearce
	 *
	 */
	public static class Token {

		public enum Kind {
			Identifier,
			// Keywords
			True { public String toString() { return "true"; }},
			False { public String toString() { return "true"; }},
			Null { public String toString() { return "null"; }},
			Void { public String toString() { return "void"; }},
			Bool { public String toString() { return "bool"; }},
			Int { public String toString() { return "int"; }},
			Real { public String toString() { return "real"; }},
			Char { public String toString() { return "char"; }},
			String { public String toString() { return "string"; }},
			If { public String toString() { return "if"; }},
			Switch { public String toString() { return "switch"; }},
			While { public String toString() { return "while"; }},
			Else { public String toString() { return "else"; }},
			Is { public String toString() { return "is"; }},
			In { public String toString() { return "in"; }},
			For { public String toString() { return "for"; }},
			Debug { public String toString() { return "debug"; }},
			Print { public String toString() { return "print"; }},
			Return { public String toString() { return "return"; }},
			Constant { public String toString() { return "constant"; }},
			Type { public String toString() { return "type"; }},
			// Constants (Given a toString for error handling purposes)
			RealValue { public String toString() { return "real"; }},
			IntValue { public String toString() { return "int"; }},
			CharValue { public String toString() { return "char"; }},
			StringValue { public String toString() { return "string"; }},
			// Symbols
			Comma { public String toString() { return ","; }},
			SemiColon { public String toString() { return ";"; }},
			Colon { public String toString() { return ":"; }},
			VerticalBar { public String toString() { return "|"; }},
			LeftBrace { public String toString() { return "("; }},
			RightBrace { public String toString() { return ")"; }},
			LeftSquare { public String toString() { return "["; }},
			RightSquare { public String toString() { return "]"; }},
			LeftAngle { public String toString() { return "<"; }},
			RightAngle { public String toString() { return ">"; }},
			LeftCurly { public String toString() { return "{"; }},
			RightCurly { public String toString() { return "}"; }},
			PlusPlus { public String toString() { return "++"; }},
			Plus { public String toString() { return "+"; }},
			Minus { public String toString() { return "-"; }},
			Star { public String toString() { return "*"; }},
			LeftSlash { public String toString() { return "\\"; }},
			RightSlash { public String toString() { return "//"; }},
			Percent { public String toString() { return "%"; }},
			Shreak { public String toString() { return "!"; }},
			Dot { public String toString() { return "."; }},
			DotDot { public String toString() { return ".."; }},
			Equals { public String toString() { return "="; }},
			EqualsEquals { public String toString() { return "=="; }},
			NotEquals { public String toString() { return "!="; }},
			LessEquals { public String toString() { return "<="; }},
			GreaterEquals { public String toString() { return ">="; }},
			LogicalAnd { public String toString() { return "&&"; }},
			LogicalOr { public String toString() { return "||"; }},
			// Other
			NewLine,
			Indent,

			//Used by error handler to identify expected types
			ExprLval { public String toString() { return "variable, list access, or record access"; }},
			Expression { public String toString() { return "<<Expression>>"; }},
			Statement { public String toString() { return "<<Statement>>"; }},
			Type2 { public String toString() { return "<<Type>>"; }}
		}

		public final Kind kind;
		public final String text;
		public final int start;

		public Token(Kind kind, String text, int pos) {
			this.kind = kind;
			this.text = text;
			this.start = pos;
		}

		public int end() {
			return start + text.length() - 1;
		}
	}
}
