package wyscript.par.util;

import java.util.ArrayList;
import java.util.List;

import wyscript.lang.Stmt;
import wyscript.lang.Stmt.ParFor;
import wyscript.par.loop.GPULoop;
import wyscript.par.loop.GPUNestedLoop;

/**
 * LoopFilter is responsible for identifying parts of a ParFor
 * loop that can run on the GPU.
 * @author antunomate
 *
 */
public class LoopFilterFactory {
	private LoopFilterFactory(){}

	public static GPULoop produceLoop(Stmt.ParFor loop) {
		Category category = classify(loop);
		switch (category) {
		case CPU:
			return null; //TODO add error here
		case GPUEXPLICITNESTED:
			return new GPUNestedLoop(loop);
		case GPUEXPLICITNONNESTED:
			return new GPUNestedLoop(loop);
		case GPUIMPLICITNESTED:
			return new GPUNestedLoop(loop);
		default:
			return null; //TODO add error here
		}
	}
	/**
	 * Classifies loops into one of three categories.
	 * This is computed in a very simple fashion, without complicated analysis.
	 * @param loop
	 * @return
	 */
	public static Category classify(Stmt.ParFor loop) {
		Category category = null;
		if (loop.getBody().size() == 1) {
			Stmt stmt = loop.getBody().get(0);
			//this loop may be implicit
			if (stmt instanceof Stmt.ParFor) {
				category = classify((ParFor) stmt);
				if (category == Category.GPUIMPLICITNESTED) {
					//then this loop is GPU-implicit
					return Category.GPUEXPLICITNONNESTED;
				}
			}else {
				if (stmt instanceof Stmt.For) {
					return Category.GPUEXPLICITNESTED;
				}else {
					return Category.GPUIMPLICITNESTED;
				}
			}
			return category;
		}
		else {
			for (Stmt stmt : loop.getBody()) {
				if (stmt instanceof Stmt.For) {
					category = Category.GPUEXPLICITNESTED;
				}
			}
			return category;
		}
	}
}
