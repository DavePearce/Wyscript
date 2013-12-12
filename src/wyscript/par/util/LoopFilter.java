package wyscript.par.util;

import java.util.ArrayList;
import java.util.List;

import wyscript.lang.Stmt;
import wyscript.lang.Stmt.ParFor;

/**
 * LoopFilter is responsible for identifying parts of a ParFor
 * loop that can run on the GPU.
 * @author antunomate
 *
 */
public class LoopFilter {
	List<Stmt> result = new ArrayList<Stmt>();
	List<LoopModule> modules;

	public enum Cat {
		IMP,
		IMPINNER,
		EXP,
		CPU
	}
	/**
	 * Classifies loops into one of three categories.
	 * This is computed in a very simple fashion, without complicated analysis.
	 * @param loop
	 * @return
	 */
	public static Cat classify(Stmt.ParFor loop) {
		Cat category = null;
		if (loop.getBody().size() == 1) {
			Stmt stmt = loop.getBody().get(0);
			//this loop may be implicit
			if (stmt instanceof Stmt.ParFor) {
				category = classify((ParFor) stmt);
				if (category == Cat.IMP) {
					//then this loop is GPU-implicit
					return Cat.IMPINNER;
				}
			}else {
				if (stmt instanceof Stmt.For) {
					return Cat.EXP;
				}else {
					return Cat.IMP;
				}
			}
			return category;
		}
		else {
			for (Stmt stmt : loop.getBody()) {
				if (stmt instanceof Stmt.For) {
					category = Cat.EXP;
				}
			}
			return category;
		}
	}
}
