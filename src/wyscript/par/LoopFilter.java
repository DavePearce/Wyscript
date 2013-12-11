package wyscript.par;

import java.util.ArrayList;
import java.util.List;

import wyscript.lang.Stmt;

/**
 * LoopFilter is responsible for identifying parts of a ParFor
 * loop that can run on the GPU.
 * @author antunomate
 *
 */
public class LoopFilter {
	List<Stmt> result = new ArrayList<Stmt>();

	public Stmt filter(Stmt.ParFor loop , int depth) {
		for (Stmt statement : loop.getBody()) {
			//what do we want to happen here?
			if (statement instanceof Stmt.ParFor) {
				//nested gpu loop potentially
				Stmt filtered = filter(loop,(Stmt.ParFor)statement,1);
				return filtered;
			}else {
				return statement;
			}
		}
	}
	public Stmt filter(Stmt.ParFor parent , Stmt.ParFor loop , int depth) {
		for (Stmt statement : loop.getBody()) {
			//what do we want to happen here?
			if (statement instanceof Stmt.ParFor) {
				//nested gpu loop potentially
				filter()
			}
		}
		return result;
	}
}
