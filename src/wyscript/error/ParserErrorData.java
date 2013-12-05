package wyscript.error;

import java.util.List;
import wyscript.io.Lexer.Token;

/**
 * Simple construct holding all the information the ErrorHandler requires about an
 * error that occurred during the parsing stage of compilation.
 *
 * @author Daniel Campbell
 *
 */
public class ParserErrorData implements ErrorData{

	private List<String> methodTrace;	//The method call stack leading to this error
	private Token found;				//The token that generated this error (null if EOF)
	private Token expected;				//The expected token (null if at a decision point)
	private ErrorType type;				//The type of error that occurred

	public ParserErrorData(List<String> m, Token f, Token e, ErrorType t) {
		methodTrace = m;
		found = f;
		expected = e;
		type = t;
	}

	public List<String> methods() {
		return methodTrace;
	}

	public Token found() {
		return found;
	}

	public Token expected() {
		return expected;
	}

	public ErrorType type() {
		return type;
	}

	/**
	 * The three main forms of error:
	 *
	 * - a token was expected but not found
	 * - a token was found in place of another token (in the middle of a parse)
	 * - at a decision point in the parser, a token was found that was invalid for the decision
	 */
	public static enum ErrorType {
		MISSING_TOKEN, UNEXPECTED_TOKEN, INVALID_TOKEN
	}
}
