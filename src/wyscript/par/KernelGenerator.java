package wyscript.par;

/**
 * KernelGenerator provides a set of utilities that activate parallel-for loops
 * so that they can run on the GPU.
 */
import static wyscript.util.SyntaxError.syntaxError;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.BoundCalc;
import wyscript.lang.Type;
import wyscript.lang.WyscriptFile;
import wyscript.lang.WyscriptFile.FunDecl;
import wyscript.lang.WyscriptFile.Parameter;
import wyscript.par.util.GPULoop;
import wyscript.par.util.LoopModule;
import wyscript.util.SyntaxError.InternalFailure;
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
	 * @param file
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
		generateKernels(functions,wyFile);
	}
	/**
	 * Scan an individual function for loops that can be converted
	 * @param functions
	 * @param file
	 * @param environment
	 */
	public static void generateKernels(Map<String, FunDecl> functions, WyscriptFile file) {
		for (String fname : functions.keySet()) {
			WyscriptFile.FunDecl func = functions.get(fname);
			//pass the name of the function down so it can be used to address kernel
			scanFuncBody(func, func.statements , fname , file);
		}
	}
	/**
	 * Scans the function body and identifies ParFor loops. Recursively scans
	 * block statements.
	 * @param function
	 * @param funcname
	 * @param environment
	 */
	public static void scanFuncBody(WyscriptFile.FunDecl function ,
			List<Stmt> statements , String funcname , WyscriptFile file) {
		int loopPosition = 0;
		//Map<String, Type> environment = new HashMap<String,Type>();
		//checker.check(function.statements , environment);
		Map<String,Type> env = new HashMap<String,Type>();
		for (int i= 0; i < statements.size() ; i++) {
			Stmt statement = statements.get(i);
			//update the environment
			env = getEnvironment(statement, file , env);
			if (statement instanceof Stmt.ParFor) {
				Stmt.ParFor loop = (Stmt.ParFor) statement;
				HashMap<String,Type> newEnv = new HashMap<String,Type>(env);
				String filename = funcname + Integer.toString(loopPosition);
				KernelRunner runner = generateForKernel((Stmt.ParFor)statement
						, newEnv , filename);
				//now plug the kernel runner into the loop
				loop.setKernelRunner(runner);
				loopPosition++;
			}else if (statement instanceof Stmt.While) {
				Map<String,Type> newEnv = new HashMap<String,Type>(env);
				scanBody(function, newEnv, ((Stmt.While) statement).getBody(), funcname, file, loopPosition);
			}
		}
	}
	public static void scanBody(WyscriptFile.FunDecl function , Map<String,Type> env ,
			List<Stmt> statements , String funcname , WyscriptFile file, int loopPosition) {
		for ( Parameter param : function.parameters) {
			env.put(param.name, param.type);
		}
		for (int i= 0; i < statements.size() ; i++) {
			Stmt statement = statements.get(i);
			//update the environment
			env = getEnvironment(statement, file , env);
			if (statement instanceof Stmt.ParFor) {
				Stmt.ParFor loop = (Stmt.ParFor) statement;
				HashMap<String,Type> newEnv = new HashMap<String,Type>(env);
				String filename = funcname + Integer.toString(loopPosition); //TODO make loop positions unique.
				KernelRunner runner = generateForKernel((Stmt.ParFor)statement
						, newEnv , filename);
				//now plug the kernel runner into the loop
				loop.setKernelRunner(runner);
				loopPosition++;
			}else if (statement instanceof Stmt.While) {
				scanBody(function, env, ((Stmt.While) statement).getBody(), funcname, file, loopPosition);
			}
		}
	}
	/**
	 * Returns a KernelRunner for this loop.
	 * @param loop The loop to be converted for running on GPU
	 * @param environment The type environment so far
	 * @param filename The file name under which the loop is invoked
	 * @return
	 */
	public static KernelRunner generateForKernel(Stmt.ParFor loop ,
			HashMap<String,Type> environment , String filename) {
		//writer = new KernelWriter(id, environment, loop);
		GPULoop gpuLoop = new GPULoop(loop);
		gpuLoop.initialiseArguments(environment);
		LoopModule module = new LoopModule(filename, environment, gpuLoop);
		KernelWriter writer = new KernelWriter(module);
		KernelRunner runner = new KernelRunner(writer.getPtxFile(),module);
		return runner;
	}
	/**
	 * Returns the entire environment of the function
	 * @param function
	 * @param file
	 * @return
	 */
	public static Map<String,Type> getEnvironment(Stmt stmt, WyscriptFile file,
			Map<String,Type> env) {
		if (stmt instanceof Stmt.VariableDeclaration) {
				getEnvironment((Stmt.VariableDeclaration)stmt,env,file);
		}
		return env;
	}

	public static void getEnvironment(Stmt.VariableDeclaration stmt, Map<String, Type>
	environment, WyscriptFile file) {
		Expr expression = stmt.getExpr();
		String name = stmt.getName();
		environment.put(name, stmt.getType());
	}
}
