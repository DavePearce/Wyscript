package wyscript.par.loop;

import java.util.HashMap;
import java.util.List;

import wyscript.lang.Stmt.ParFor;
import wyscript.par.util.Argument;

public class GPUNestedLoop extends GPULoop{

	public GPUNestedLoop(ParFor loop) {
		super(loop);
		// TODO Auto-generated constructor stub
	}

	@Override
	public ParFor getLoop() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<Argument> getArguments() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int innerLowerBound(HashMap<String, Object> frame) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int innerUpperBound(HashMap<String, Object> frame) {
		// TODO Auto-generated method stub
		return 0;
	}

}
