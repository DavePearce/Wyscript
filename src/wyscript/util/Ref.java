package wyscript.util;

/**
 * Used in the interpreter to represent an object reference
 * created with the new expression.
 */
public class Ref {

	private Object value;

	public Ref(Object value) {
		this.value = value;
	}

	public Object getValue() {
		return value;
	}

	public void setValue(Object v) {
		value = v;
	}

	public String toString() {
		return "&" + value;
	}
}
