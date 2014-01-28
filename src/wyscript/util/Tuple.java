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

	public boolean equals(Object o) {
		if (!(o instanceof Tuple))
			return false;

		Tuple t = (Tuple) o;
		if (values.size() != t.values.size())
			return false;
		for (int i = 0; i < values.size(); i++) {
			if (values.get(i) == null && t.values.get(i) != null)
				return false;
			if (values.get(i) != null && t.values.get(i) == null)
				return false;
			if (!(values.get(i).equals(t.values.get(i))))
				return false;
		}
		return true;
	}

}
