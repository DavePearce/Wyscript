package wyscript.par;

//TODO consider wether this class is necessary. if not too many responsibilities,remove it!

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import wyscript.lang.Stmt;
import wyscript.lang.Stmt.For;
import wyscript.lang.Type;
import wyscript.lang.WyscriptFile;
import wyscript.lang.WyscriptFile.FunDecl;
import wyscript.util.TypeChecker;
import wyscript.util.SyntaxError.InternalFailure;

/**
 * The kernel generator is responsible for the generation Cuda kernels
 * from parallel-for loops. It maps each parallel for loop to a KernelRunner.
 * This runner is then attached to the parallel-for loop.
 * @author Mate Antunovic
 *
 */
public class KernelGenerator {
	private Map<String , WyscriptFile.FunDecl> functions;
	private Map<Stmt.ParFor,KernelRunner> forToRunner = new HashMap<Stmt.ParFor,
			KernelRunner>();
	private WyscriptFile file;
	/**
	 * Scan the Wyscript file and convert available loops to parallel loops
	 * @param wyFile
	 */
	public KernelGenerator(WyscriptFile wyFile) {
		this.file = wyFile;
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
	/**
	 * Scans the function body and identifies parallel for loops
	 * @param function
	 * @param name
	 */
	private void scanFuncBody(WyscriptFile.FunDecl function , String name) {
		int loopPosition = 0;
		Map<String,Type> environment = new HashMap<String,Type>();
		TypeChecker checker = new TypeChecker();
		//generate environment
		checker.check(function.statements , environment);
		//now have acquired type information --hopefully!
		for (int i= 0; i < function.statements.size() ; i++) {
			Stmt statement = function.statements.get(i);
//			addToEnvironment(statement , environment);
			if (statement instanceof Stmt.ParFor) {
				//this loop can be converted
				String id = name + Integer.toString(loopPosition);
				KernelRunner runner = generateForKernel((Stmt.ParFor)statement
						, environment, id);
				//note that a new hashmap is generated to protect the elements
				loopPosition++;
				forToRunner.put((Stmt.ParFor)statement,runner);
			}
		}
	}
	/**
	 * Returns a KernelRunner for this loop.
	 * @param loop The loop to be converted for running on GPU
	 * @param environment The type environment so far
	 * @param id The file name under which the loop is invoked
	 * @return
	 */
	private KernelRunner generateForKernel(Stmt.ParFor loop , Map<String,Type> environment , String id) {
		KernelWriter writer = null;
		writer = new KernelWriter(id, environment, loop);
		return writer.getRunner();
	}

}
