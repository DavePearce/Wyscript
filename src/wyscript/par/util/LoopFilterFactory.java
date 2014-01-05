package wyscript.par.util;

import java.util.ArrayList;
import java.util.List;

import wyscript.lang.Stmt;
import wyscript.lang.Stmt.ParFor;
import wyscript.par.loop.GPULoop;
import wyscript.par.loop.GPUNestedLoopExplicit;
import wyscript.par.loop.GPUNestedLoopImplicit;
import wyscript.par.loop.GPUSingleLoop;
import wyscript.util.SyntaxError.InternalFailure;

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
			InternalFailure.internalFailure("Cannot parallelise loop", "", loop);
		case GPU_PART_IMPLICIT_NESTED:
			return new GPUNestedLoopImplicit(loop);
		case GPU_PART_IMPLICIT_NONNESTED:
			return new GPUSingleLoop(loop);
		case GPU_IMPLICIT_NONNESTED:
			return new GPUSingleLoop(loop);
		case GPU_NOTALOOP:
			return new GPUSingleLoop(loop);
		case GPU_EXPLICIT_NESTED:
			return new GPUNestedLoopExplicit(loop);
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
		Category category = Category.GPU_IMPLICIT_NONNESTED;
		if (loop.getBody().size() == 1) {
			Stmt stmt = loop.getBody().get(0);
			//this loop may be implicit
			if (stmt instanceof Stmt.ParFor) {
				category = classify((ParFor) stmt);
				if (category == Category.GPU_IMPLICIT_NONNESTED) {
					//then this loop is GPU-implicit
					return Category.GPU_PART_IMPLICIT_NESTED;
				}
			}else {
				return Category.GPU_IMPLICIT_NONNESTED;
			}
		}
		else {
			for (Stmt stmt : loop.getBody()) {
				if (stmt instanceof Stmt.For) {
//					category = Category.GPU_EXPLICIT_NESTED;
//					return category;
				}else if (stmt instanceof Stmt.ParFor) {
					Category nestedCat = classify((ParFor) stmt);
					switch (category) {
					case CPU:
						return Category.CPU;
					case GPU_PART_IMPLICIT_NESTED:
						return Category.GPU_PART_IMPLICIT_NESTED;
					case GPU_PART_IMPLICIT_NONNESTED:
						return Category.GPU_PART_IMPLICIT_NESTED;
					case GPU_IMPLICIT_NESTED:
						return Category.GPU_PART_IMPLICIT_NESTED;
					case GPU_IMPLICIT_NONNESTED:
						return Category.GPU_PART_IMPLICIT_NESTED;
					case GPU_NOTALOOP:
						return Category.GPU_IMPLICIT_NONNESTED;
					default:
						return null; //TODO put error here
					}
				}
			}
		}
		return category;
	}
}
