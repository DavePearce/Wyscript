package wyscript.par;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import wyscript.lang.Stmt;
import wyscript.lang.Stmt.For;
import wyscript.lang.WyscriptFile;
import wyscript.lang.WyscriptFile.FunDecl;

/**
 * The kernel generator is responsible for the generation Cuda kernels
 * from parallel for loops.
 * @author antunomate
 *
 */
public class KernelGenerator {
	private Map<String , WyscriptFile.FunDecl> functions;
	/**
	 * Scan the Wyscript file and convert available loops to parallel loops
	 * @param wyFile
	 */
	public KernelGenerator(WyscriptFile wyFile) {
		for (WyscriptFile.Decl declaration : wyFile.declarations) {
			if(declaration instanceof WyscriptFile.FunDecl) {
				WyscriptFile.FunDecl fd = (WyscriptFile.FunDecl) declaration;
				this.functions.put(fd.name(), fd);
			}
		}
		generateKernels(functions);
	}
	/**
	 * Scan an individual function for loops that can be converted
	 * @param functions
	 */
	private void generateKernels(Map<String, FunDecl> functions) {
		for (String fname : functions.keySet()) {
			WyscriptFile.FunDecl func = functions.get(fname);
			//pass the name of the function down so it can be used to address kernel
			scanFuncBody(func , fname);
		}
	}

	private void scanFuncBody(WyscriptFile.FunDecl function , String name) {
		int loopPosition = 0;
		for (int i= 0; i < function.statements.size() ; i++) {
			Stmt statement = function.statements.get(i);
			if (statement instanceof Stmt.ParFor) {
				//this loop can be converted
				String id = name + Integer.toString(loopPosition);
				KernelRunner runner = generateForKernel((Stmt.ParFor)statement , loopPosition);
				loopPosition++;
			}
		}
	}
	/**
	 * Returns a kernel runner for this loop
	 * @param loop
	 * @param loop
	 * @return
	 */
	private KernelRunner generateForKernel(Stmt.ParFor loop) {
		KernelWriter writer = new KernelWriter(loop);
		return null;

	}

}
