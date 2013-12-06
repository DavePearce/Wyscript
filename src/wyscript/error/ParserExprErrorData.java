package wyscript.error;

import wyscript.io.Lexer.Token;
import wyscript.io.Lexer.Token.Kind;
import wyscript.lang.Expr;

/**
 * Extension to the ParserErrorData that stores a whole expression instead of a single token
 *
 * @author Daniel Campbell
 *
 */
public class ParserExprErrorData extends ParserErrorData {

	private Expr expr;

	public ParserExprErrorData(String n, Expr ex, Token f, Kind e, int s, int d, ErrorType t) {
		super(n, f, e, s, d, t);
		expr = ex;
	}

	public Expr expr() {
		return expr;
	}

}
