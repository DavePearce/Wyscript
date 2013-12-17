package wyscript.par.util;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.par.KernelRunner;
import wyscript.par.KernelWriter;
import wyscript.par.loop.GPULoop;
import wyscript.util.SyntaxError.InternalFailure;

public class LoopModule {
	private Map<String , Type> environment; //passed to kernel writer at runtime

	private String fileName;
	private GPULoop gpuLoop;

	/**
	 * Initialise a KernelWriter which takes <i>name<i/> as its file name and uses
	 * the type mapping given in <i>environment</i> to generate the appropriate kernel
	 * for <i>loop</i>.
	 * @param filename
	 * @param environment
	 * @param loop
	 * @throws IOException
	 * @requires A correct mapping of the symbols used (when the parFor is executed) to their types
	 * @ensures All necessary parameters extracted and converted into a Cuda kernel, as well as stored within KernelWriter
	 */
	public LoopModule(String filename , Map<String , Type> environment , GPULoop loop){
		this.environment = environment;
		this.fileName = filename;
		this.gpuLoop = loop;
	}
	public String getName() {
		return this.fileName;
	}
	public Map<String,Type> getEnvironment() {
		return new HashMap<String,Type>(environment);
	}

	public GPULoop getGPULoop() {
		return gpuLoop;
	}
}
