package wyscript.error;

import wyscript.io.Lexer.Token;

/**
 * Simple construct holding all the information the ErrorHandler requires about an
 * error that occurred during the parsing stage of compilation.
 *
 * @author Daniel Campbell
 *
 */
public class ParserErrorData implements ErrorData{

	private String filename;			//The file this error occurred in
	private Token found;				//The token that generated this error (null if EOF)
	private Token.Kind expected;		//The expected token type (null if at a decision point)
	private ErrorType type;				//The type of error that occurred
	private int start;
	private int end;

	public ParserErrorData(String n, Token f, Token.Kind e, ErrorType t) {
		filename = n;
		found = f;
		expected = e;
		type = t;
		start = f.start;
		end = f.end();
	}

	public ParserErrorData(String n, Token f, Token.Kind e, int s,
			int d, ErrorType t) {
		filename = n;
		found = f;
		expected = e;
		type = t;
		start = s;
		end = d;
	}

	public String filename() {
		return filename;
	}

	public Token found() {
		return found;
	}

	public Token.Kind expected() {
		return expected;
	}

	public int start() {
		return start;
	}

	public int end() {
		return end;
	}

	public ErrorType type() {
		return type;
	}

	/**
	 * The different types of error:
	 *
	 * - a token was expected but not found
	 * - the indent was increased within a block
	 * - An expression was found, but not of the required type
	 * - An expression was expected but not found
	 * - A duplicate name was used when creating a record
	 * - The parser tried to parse a type, but couldn't
	 */
	public static enum ErrorType {
		MISSING_TOKEN, BAD_INDENT, BAD_EXPRESSION_TYPE, MISSING_EXPRESSION, DUPLICATE_TOKEN, INVALID_TYPE,
		BAD_SWITCH_CONST, DUPLICATE_SWITCH_CONST, BAD_SWITCH_CASE, SWITCH_MULTIPLE_DEFAULT, BAD_INCLUDE,
		NAME_CLASH
	}
}
