package wyscript.par.loop;


import java.util.HashMap;

import wyscript.lang.Stmt;
import wyscript.lang.Stmt.ParFor;

public class GPUSingleLoop extends GPULoop {
	private final Stmt.ParFor loop;

	public GPUSingleLoop(Stmt.ParFor loop) {
		super(loop);
		this.loop = loop;
	}
	public ParFor getLoop() {
		return this.loop;
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
