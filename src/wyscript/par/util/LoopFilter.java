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
				if (category == Category.IMP) {
					//then this loop is GPU-implicit
					return Category.IMPINNER;
				}
			}else {
				if (stmt instanceof Stmt.For) {
					return Category.EXP;
				}else {
					return Category.IMP;
				}
			}
			return category;
		}
		else {
			for (Stmt stmt : loop.getBody()) {
				if (stmt instanceof Stmt.For) {
					category = Category.EXP;
				}
			}
			return category;
		}
	}
}
