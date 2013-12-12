package wyscript.par;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import jcuda.*;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;
import wyscript.Interpreter;
import wyscript.lang.*;
import wyscript.lang.Expr.BOp;
import wyscript.lang.Expr.Binary;
import wyscript.par.util.Argument;
import wyscript.par.util.LoopModule;
import wyscript.util.SyntaxError.InternalFailure;
import static jcuda.driver.CUresult.*;
import static jcuda.runtime.JCuda.*;

/**
 * What might the kernel runner do? It will require some data structures to
 * marshall between Cuda and Wyscript. It also needs to be extensible to allow
 * different types of loop runs. Maybe the kernel writer is associated with
 * kernel runner...
 *
 * @author Mate Antunovic
 *
 */
public class KernelRunner {
	private CUfunction function;
	private Interpreter interpreter;
	//Three lists used to track the name, type and device pointer of elements

	private List<CUdeviceptr> devicePointers;

	private List<Argument> arguments;

	private File file;

	private int blockSizeX = 16;
	private CUcontext context;
	private LoopModule module;
	public KernelRunner(LoopModule module) {
		this.module = module;
		this.interpreter = new Interpreter();
		file = module.getPtxFile();
		arguments = module.getArguments();
		devicePointers = new ArrayList<CUdeviceptr>();
		initialise();
	}

	/**
	 * This will initialise the kernel from the given file name
	 *
	 * @param ptxFileName
	 */
	public void initialise() {
		String funcname = module.getFuncName();
		String ptxFileName = file.getAbsolutePath();
		int result;
		// initialise driver and create context
		result = cuInit(0);
		stopIfFailed(result);
		CUdevice device = new CUdevice();
		result = cuDeviceGet(device, 0);
		stopIfFailed(result);
		CUcontext context = new CUcontext();
		//create device context
		result = cuCtxCreate(context, 0, device);
		this.context = context;
		stopIfFailed(result);
		// Load the ptx file.
		CUmodule module = new CUmodule();
		result = cuModuleLoad(module, ptxFileName);
		stopIfFailed(result);
		// Obtain a function pointer to the correct function function.
		CUfunction function = new CUfunction();
		result = cuModuleGetFunction(function, module, funcname);
		stopIfFailed(result);
		this.function = function;
		// host input data ready to be filled
	}
	/**
	 * Helper method to stop program if there is a cuda-related error
	 * @param result The cuda error code (or success code)
	 */
	private void stopIfFailed(int result) {
		if (result != CUDA_SUCCESS) {
			Thread.dumpStack();
			InternalFailure.internalFailure(
					"Failure with code"+result+". Got error: "+
			cudaGetErrorString(result), file.getName(), module.getOuterLoop());
		}
	}

	/**
	 * This runs the kernel passing the values to the function
	 *
	 * @param interpreter
	 * @param frame
	 * @param values
	 *            -- the RHS values
	 *
	 * @requires All the types within the type list match up correctly with the
	 *           Cuda kernel
	 * @ensures The loop runs on the kernel and the values of the computation
	 *          placed on frame once completed
	 */
	public Object run(HashMap<String, Object> frame) {
		List<CUdeviceptr> pointers = marshalParametersToGPU(frame);
		NativePointerObject[] parametersPointer = getPointerToParams(pointers);
		int gridDimX = getLoopRange(module.getOuterLoop(), frame);
		Stmt.ParFor innerLoop = module.getInnerLoop();
		int gridDimY;
		if (innerLoop != null) {
			gridDimY = getLoopRange(innerLoop, frame);
		}else {
			gridDimY = 1;
		}
		int result = cuLaunchKernel(function,
				gridDimX, gridDimY, 1,
				blockSizeX, 1, 1,
				0, null,
				Pointer.to(parametersPointer), null);
		int syncResult = cuCtxSynchronize();
		stopIfFailed(syncResult);
		if (result != CUDA_SUCCESS) {
			InternalFailure.internalFailure("Kernel did not launch successfully." +
					cudaGetErrorString(result), file.getName(), module.getOuterLoop());
		}
		marshalParametersFromGPU(frame,pointers);
		//cuCtxDestroy(context); //major error avoided by commenting this line
		cleanUp(pointers);
		return null; //no need to return any particular object
	}
	private int getLoopRange(Stmt.ParFor loop , HashMap<String,Object> frame) {
		Expr src = loop.getSource();
		if (src instanceof Expr.Binary) {
			Expr.Binary binary = (Binary) src;
			Expr.BOp op = binary.getOp();
			if (op.equals(BOp.RANGE)) {
				try {
					Integer low = (Integer)evaluate(binary.getLhs(), frame);
					Integer high = (Integer)evaluate(binary.getRhs(), frame);
					return high - low;
				}catch (ClassCastException e) {
					InternalFailure.internalFailure("Attempted to read range of binary expression." +
							"Could not cast to integer", file.getPath(), src);
				}
			}
		}
		return 0; //unreachable code
	}
	/**
	 * Fill me in with some code that was used elsewhere to access expression
	 * values
	 * @param expression
	 * @param frame
	 * @return
	 */
	private Object evaluate(Expr expression , HashMap<String,Object> frame) {
		return interpreter.execute(expression,frame);
	}

	/**
	 * Frees the list of device pointers
	 * @param pointers
	 */
	private void cleanUp(List<CUdeviceptr> pointers) {
		for (CUdeviceptr ptr : pointers) {
			cuMemFree(ptr);
		}
	}

	/**
	 * Returns an array of pointers to device pointers
	 * @param devicePointers
	 * @return
	 */
	private NativePointerObject[] getPointerToParams(List<CUdeviceptr> devicePointers) {
		int length = devicePointers.size();
		Pointer[] devicePointerArray = new Pointer[length];
		for (int i = 0 ; i < devicePointerArray.length ; i++) {
			devicePointerArray[i] = Pointer.to(devicePointers.get(i));
		}
		return devicePointerArray;
	}

	/**
	 * Responsible for converting data back from the GPU to the format
	 * expected on the frame
	 * @param frame
	 */
	private void marshalParametersFromGPU(HashMap<String, Object> frame,List<CUdeviceptr> devicePointers) {
		for (int i = 0 ; i < arguments.size() ; i++) {
			CUdeviceptr ptr = devicePointers.get(i);
			//update the frame
			arguments.get(i).read(frame, ptr);
		}
	}
	/**
	 * Marshalls the data from the parameters into an appropriate format and
	 * allocates memory appropriately, filling in pointers for kernel parameters.
	 * This method is called to upload data onto the GPU.
	 * @param frame
	 * @return
	 *
	 */
	private List<CUdeviceptr> marshalParametersToGPU(HashMap<String, Object> frame) {
		List<CUdeviceptr> devPtrs = new ArrayList<CUdeviceptr>(arguments.size());
		for (int i = 0 ; i < arguments.size() ; i++) {
			Argument arg = arguments.get(i);
			CUdeviceptr ptr = new CUdeviceptr();
			arg.write(frame, ptr);
			devPtrs.add(ptr);
		}
		return devPtrs;
	}
	/**
	 * Allocates device memory for the integer array and returns a pointer to
	 * it.
	 * @param intArray
	 * @return
	 */
	private CUdeviceptr generateH2DPointer(int[] intArray) {
		CUdeviceptr dpointer = new CUdeviceptr();
		int expectedLength = intArray.length;
		cuMemAlloc(dpointer, expectedLength * Sizeof.INT);
		cuMemcpyHtoD(dpointer, Pointer.to(intArray), expectedLength
				* Sizeof.INT);
		return dpointer;
	}

}
