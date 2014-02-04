package wyscript.error;

import wyscript.lang.Expr;
import wyscript.util.Attribute;
import wyscript.util.SyntacticElement;

/**
 * Holds data about an error that occurred within the Type Checker - used
 * for error recovery and for making code suggestions.
 *
 * @author Daniel Campbell
 *
 */
public class TypeErrorData implements ErrorData {

	private String filename;
	private  Expr found;
	private  SyntacticElement expected;
	private  int start, end;
	private  ErrorType type;

	public TypeErrorData(String filename, Expr found, SyntacticElement expected, Attribute.Source source, ErrorType type) {
		this.filename = filename;
		this.found = found;
		this.expected = expected;
		this.start = source.start;
		this.end = source.end;
		this.type = type;
	}

	public String filename() {
		return filename;
	}

	public Expr found() {
		return found;
	}

	public SyntacticElement expected() {
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
	 * Represents all the possible types of error:
	 * - non-void function missing return type
	 * - duplicate variable declaration
	 * - for loop declared with non-list expression
	 * - Switch expression is a record type (which is invalid)
	 * - Next statement outside of a switch
	 * - Called a function that didn't exist
	 * - Incorrect number of function arguments
	 * - Tried to access field of non-record type
	 * - Tried to access field that doesn't exist
	 * - Expected one type, found another
	 * - Expected one type to be a subtype of another
	 * - Referenced an undeclared variable
	 * - Assigned to a tuple that contained expressions that couldn't be assigned to
	 * - Tried to cast a reference type
	 */
	public static enum ErrorType {
		MISSING_RETURN, DUPLICATE_VARIABLE, BAD_FOR_LIST, BAD_SWITCH_TYPE, BAD_NEXT, MISSING_FUNCTION,
		BAD_FUNC_PARAMS, BAD_FIELD_ACCESS, MISSING_FIELD, TYPE_MISMATCH, SUBTYPE_MISMATCH, UNDECLARED_VARIABLE,
		BAD_TUPLE_ASSIGN, BAD_REFERENCE_CAST
	}
}
