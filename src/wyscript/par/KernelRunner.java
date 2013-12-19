package wyscript.par;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import jcuda.*;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;
import wyscript.par.loop.GPULoop;
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
	private List<CUdeviceptr> devicePointers = new ArrayList<CUdeviceptr>();
	private File file;

	private LoopModule module;
	private GPULoop gpuLoop;

	private int blockDimX = 16; //garanteed to be set later
	private int blockDimY = 16;
	private int blockDimZ = 1;

	private int gridDimX = 10;
	private int gridDimY = 10;
	private int gridDimZ = 1;

	public KernelRunner(File ptxFile , LoopModule module) {
		this.module = module;
		this.gpuLoop = module.getGPULoop();
		file = ptxFile;
		devicePointers = new ArrayList<CUdeviceptr>();
		initialise();
	}

	/**
	 * This will initialise the kernel from the given file name
	 *
	 * @param ptxFileName
	 */
	public void initialise() {
		String funcname = module.getName();
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
			cudaGetErrorString(result), file.getName(), gpuLoop.getLoop());
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
		computeDimensions(frame);
		int result = cuLaunchKernel(function,
				gridDimX, gridDimY, gridDimZ,
				blockDimX, blockDimY, blockDimZ,
				0, null,
				Pointer.to(parametersPointer), null);
		int syncResult = cuCtxSynchronize();
		stopIfFailed(syncResult);
		if (result != CUDA_SUCCESS) {
			InternalFailure.internalFailure("Kernel did not launch successfully." +
					cudaGetErrorString(result), file.getName(), gpuLoop.getLoop());
		}
		marshalParametersFromGPU(frame,pointers);
		return null; //no need to return any particular object
	}

	private void computeDimensions(HashMap<String, Object> frame) {
		int lowX = gpuLoop.outerLowerBound(frame);
		int highX = gpuLoop.outerUpperBound(frame);
		int lowY = gpuLoop.innerLowerBound(frame);
		int highY = gpuLoop.innerUpperBound(frame);
		//if (highY-lowY < 0)
		int diffX = Math.abs(highX-lowX);
		int diffY = Math.abs(highY-lowY);
		if (diffY == 0) {
			this.blockDimX = 256;
			this.blockDimY = 1;
			int remaining = diffX/(256)+1;
			this.gridDimX = remaining;
			this.gridDimY = 1;
			return;
		}
		else {
			int total = diffX*diffY;
			if (total >= 512) {
				this.blockDimX = 16;
				this.blockDimY = 16;
				int remaining = total;
				this.gridDimX = (int) Math.ceil(Math.sqrt(remaining))+2;
				this.gridDimY = (int) Math.ceil(Math.sqrt(remaining))+2;
			}else {
				this.blockDimX = 21;
				this.blockDimY = 21;
				this.gridDimX = 20;
				this.gridDimY = 20;
			}
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
		List<Argument> arguments = gpuLoop.getArguments();
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
		//List<CUdeviceptr> devPtrs = new ArrayList<CUdeviceptr>(arguments.size());
		//this.devicePointers = devPtrs;
		List<Argument> arguments = gpuLoop.getArguments();
		for (int i = 0 ; i < arguments.size() ; i++) {
			Argument arg = arguments.get(i);
			CUdeviceptr ptr = null;
			if (i >= devicePointers.size()) {
				ptr = new CUdeviceptr();
				devicePointers.add(ptr);
			}else {
				ptr = devicePointers.get(i);
			}
			arg.write(frame, ptr);
		}
		return devicePointers;
	}

}
