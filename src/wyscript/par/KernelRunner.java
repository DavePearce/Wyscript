package wyscript.par;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import jcuda.*;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;
import wyscript.lang.*;

/**
 * What might the kernel runner do? It will require some data structures to
 * marshall between Cuda and Wyscript. It also needs to be extensible to
 * allow different types of loop runs. Maybe the kernel writer is associated
 * with kernel runner...
 * @author antunomate
 *
 */
public class KernelRunner {
	/*
	 * I guess the required elements to the kernel runner include a list of
	 * parameters to the function, and the function pointer of course.
	 */
	private CUcontext context;
	private CUfunction function;
	List<String> params;

	List<Type> symbolTypes;
	List<Integer> symbolLength;
	private int numParams; //TODO ensure task parameter is filled with correct value
	private KernelWriter writer;

	public KernelRunner(KernelWriter writer) {
		this.writer = writer;
	}

	/**
	 * This will initialise the kernel from the given file name
	 * @param ptxFileName
	 */
	public void initialise(File ptxFile , String funcName) {
		String ptxFileName = ptxFile.getAbsolutePath();
		//initialise driver and create context
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
        this.context = context;
        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);
        // Obtain a function pointer to the correct function function.
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, funcName);
        this.function = function;
        //host input data ready to be filled
	}
	/**
	 * This runs the kernel passing the values to the function
	 * @param interpreter
	 * @param frame
	 * @param values -- the RHS values
	 */
	public void run(HashMap<String,Object> frame , Object[] values) {
		//sanity check the number of arguments -- values is the number of input objects
		List<NativePointerObject> paramPointers = new ArrayList<NativePointerObject>();
		paramPointers.add(Pointer.to(new int[]{numParams})); //add the count argument
		//sanity check the input value size
		Object[] outputs = new Object[values.length];
		if (values.length == params.size()) {
			//generate pointers to symbols one-by-one
			for (int i = 0; i < values.length ; i++) {
				CUdeviceptr deviceInput = new CUdeviceptr();
				cuMemAlloc(deviceInput , symbolLength.get(i));
				//TODO issue here is to correct types for cuda. that way we can get the pointer to the object properly

				//cuMemcpyHtoD(deviceInput , Pointer.to(values[i]), symbolLength.get(i));
			}
		}
	}
	public NativePointerObject getPointerToObject(Object value) {
		return context;
		//if (value instanceof Integer)
	}
}
