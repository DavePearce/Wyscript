package wyscript.par;

//TODO consider whether this class is necessary. if not too many responsibilities,remove it!

import java.util.HashMap;
import java.util.Map;

import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.lang.WyscriptFile;
import wyscript.lang.WyscriptFile.FunDecl;
import wyscript.util.TypeChecker;

/**
 * The kernel generator is responsible for the generation Cuda kernels
 * from parallel-for loops. It maps each parallel for loop to a KernelRunner.
 * This runner is then attached to the parallel-for loop.
 * @author Mate Antunovic
 *
 */
public class KernelGenerator {
	/**
	 * Scan the Wyscript file and convert available loops to parallel loops
	 * @param wyFile
	 */
	private KernelGenerator(){} //cannot instantiate
	/**
	 * Modifies the AST so that each ParFor loop is connected to its kernel
	 * @param wyFile
	 * @param environment
	 */
	public static void generateKernels(WyscriptFile wyFile) {
		Map<String , WyscriptFile.FunDecl> functions = new HashMap<String ,
				WyscriptFile.FunDecl>();
		for (WyscriptFile.Decl declaration : wyFile.declarations) {
			if(declaration instanceof WyscriptFile.FunDecl) {
				WyscriptFile.FunDecl fd = (WyscriptFile.FunDecl) declaration;
				functions.put(fd.name(), fd);
			}
		}
		generateKernels(functions);
	}
	/**
	 * Scan an individual function for loops that can be converted
	 * @param functions
	 * @param environment
	 */
	private static void generateKernels(Map<String, FunDecl> functions) {
		for (String fname : functions.keySet()) {
			WyscriptFile.FunDecl func = functions.get(fname);
			TypeChecker checker = new TypeChecker();
			Map<String,Type> env = new HashMap<String,Type>();
			checker.check(func.statements,env);
			//pass the name of the function down so it can be used to address kernel
			scanFuncBody(func , fname, env);
		}
	}
	/**
	 * Scans the function body and identifies parallel for loops
	 * @param function
	 * @param name
	 * @param environment
	 */
	private static void scanFuncBody(WyscriptFile.FunDecl function , String name, Map<String, Type> environment) {
		int loopPosition = 0;
		TypeChecker checker = new TypeChecker();
		//Map<String, Type> environment = new HashMap<String,Type>();
		//checker.check(function.statements , environment);
		for (int i= 0; i < function.statements.size() ; i++) {
			Stmt statement = function.statements.get(i);
			if (statement instanceof Stmt.ParFor) {
				Stmt.ParFor loop = (Stmt.ParFor) statement;
				String id = name + Integer.toString(loopPosition);
				KernelRunner runner = generateForKernel((Stmt.ParFor)statement
						, environment, id);
				//now plug the kernel runner into the loop
				loop.setKernelRunner(runner);
				loopPosition++;
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
	private static KernelRunner generateForKernel(Stmt.ParFor loop , Map<String,Type> environment , String id) {
		KernelWriter writer = null;
		writer = new KernelWriter(id, environment, loop);
		return writer.getRunner();
	}
}
