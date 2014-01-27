package wyscript.util;

import java.util.ArrayList;
import java.util.List;

/**
 * Used in the interpreter to represent a Tuple type variable.
 */
public class Tuple {

	private List<Object> values;

	public Tuple(List<Object> vals) {
		this.values = new ArrayList<Object>(vals);
	}

	public List<Object> getValues() {
		return values;
	}

	public String toString() {
		String s = "(";
		boolean first = true;
		for (Object o : values) {
			if (!first)
				s += ", ";
			first = false;
			s += o.toString();
		}
		return s + ")";
	}

}
