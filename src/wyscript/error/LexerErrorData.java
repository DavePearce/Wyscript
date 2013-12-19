package wyscript.error;

/**
 * Simple construct holding all the information the ErrorHandler requires about an
 * error that occurred during the lexing stage of compilation.
 *
 * @author Daniel Campbell
 */
public class LexerErrorData implements ErrorData {

	private int pos;				//Position in the input where error occurred
	private String filename;		//Name of the file the error occurred in
	private Character found;		//Found character (null if EOF)
	private ErrorType type;			//The type of error that occurred

	public LexerErrorData(int p, String n, Character f, ErrorType t) {
		pos = p;
		filename = n;
		found = f;
		type = t;
	}

	public int pos() {
		return pos;
	}

	public String filename() {
		return filename;
	}

	public char found() {
		return found;
	}

	public ErrorType type() {
		return type;
	}

	/**
	 * Enum representing the different possible lexer errors:
	 *
	 * - invalid/unknown character at main decision point
	 * - invalid escape character in character constant
	 * - missing closing ' on character constant
	 * - missing closing " on string constant
	 * - bad operator (probably an indicator of a compiler error, as not currently possible)
	 * - bad whitespace
	 */
	public static enum ErrorType {
		INVALID_CHARACTER, INVALID_ESCAPE, MISSING_CHAR_END, MISSING_STRING_END, INVALID_OP, INVALID_WHITESPACE,
		MISSING_COMMENT_END
	}
}
