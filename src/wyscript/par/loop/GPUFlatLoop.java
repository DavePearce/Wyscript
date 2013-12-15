package wyscript.par.loop;

import java.util.HashMap;
import java.util.List;

import wyscript.lang.Stmt;
import wyscript.lang.Stmt.ParFor;
import wyscript.par.util.Argument;

public class GPUFlatLoop extends GPULoop {
	private final Stmt.ParFor loop;

	public GPUFlatLoop(Stmt.ParFor loop) {
		super(loop);
		this.loop = loop;
		//now scan to check if this loop iterates over a range or
		//a list
	}
	public ParFor getLoop() {
		return this.loop;
	}

	public List<Argument> getArguments() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int innerLowerBound(HashMap<String, Object> frame) {
		return -1;
	}

	@Override
	public int innerUpperBound(HashMap<String, Object> frame) {
		return -1;
	}

}
