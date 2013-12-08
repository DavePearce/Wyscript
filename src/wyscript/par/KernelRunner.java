package wyscript.par;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import jcuda.*;
import jcuda.NativePointerObject;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;
import wyscript.lang.*;
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
	//the writer this kernel uses to run
	private KernelWriter writer;
	//Three lists used to track the name, type and device pointer of elements
	List<String> parameters;
	List<CUdeviceptr> devicePointers;

	File file;

	int gridDim = 16;
	int blockDim = 256;
	private CUcontext context;

	public KernelRunner(KernelWriter writer) {
		this.writer = writer;
		file = writer.getPtxFile();
		parameters = writer.getParameters();
		devicePointers = new ArrayList<CUdeviceptr>();
		initialise();
	}

	/**
	 * This will initialise the kernel from the given file name
	 *
	 * @param ptxFileName
	 */
	public void initialise() {
		String funcname = writer.getFuncName();
		String ptxFileName = file.getAbsolutePath();
		int result;
		// initialise driver and create context
		result = cuInit(0);
		stopIfFailed(result);
		CUdevice device = new CUdevice();
		result = cuDeviceGet(device, 0);
		stopIfFailed(result);
		CUcontext context = new CUcontext();

		result = cuCtxCreate(context, 0, device);
		this.context = context;
		stopIfFailed(result);
		//this.context = context;
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
		//now initialise the entire thing
	}

	private void stopIfFailed(int result) {
		if (result != CUDA_SUCCESS) {
			InternalFailure.internalFailure(
					"Failure with code"+result+". Got error: "+
			cudaGetErrorString(result), file.getName(), writer.getLoop());
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
	 *          placed on frame
	 */
	public Object run(HashMap<String, Object> frame) {
		long time = System.currentTimeMillis();
		List<CUdeviceptr> pointers = marshallParametersToGPU(frame);
		NativePointerObject[] parametersPointer = getPointerToParams(pointers);
		time = System.currentTimeMillis();
		int result = cuLaunchKernel(function,
				gridDim, 1, 1,
				blockDim, 1, 1,
				0, null,
				Pointer.to(parametersPointer), null);
		int syncResult = cuCtxSynchronize();
		//System.out.println("CALC TOOK : "+(System.currentTimeMillis()-time));
		stopIfFailed(syncResult);
		if (result != CUDA_SUCCESS) {
			//System.out.println(result);
			InternalFailure.internalFailure("Kernel did not launch successfully." +
					cudaGetErrorString(result), file.getName(), writer.getLoop());
		}
		time = System.currentTimeMillis();
		marshallParametersFromGPU(frame);
		//System.out.println("MARSHALL FROM GPU TOOK : "+(System.currentTimeMillis()-time));
		time = System.currentTimeMillis();
		cuCtxDestroy(context);
		cleanUp(pointers);
		//System.out.println("CLEANUP TOOK : "+(System.currentTimeMillis()-time));
		return null; //TODO check whether this should be changed
	}
	private void cleanUp(List<CUdeviceptr> pointers) {
		for (CUdeviceptr ptr : pointers) {
			cuMemFree(ptr);
		}
	}

	/**
	 * Returns a pointer to the parameters for the kernel
	 * @param devicePointers
	 * @return
	 */
	private NativePointerObject[] getPointerToParams(List<CUdeviceptr> devicePointers) {
		int length = devicePointers.size();
		Pointer[] devicePointerArray = new Pointer[length];
		//devicePointerArray[0] = Pointer.to(new int[]{length});
		//start off at 1 because first pointer is numparams
		for (int i = 0 ; i < devicePointerArray.length ; i++) {
			devicePointerArray[i] = Pointer.to(devicePointers.get(i));
		}
		//now take the array full of parameters and get a pointer to it
		return devicePointerArray;
	}

	/**
	 * Responsible for converting data back from the GPU to the format
	 * expected on the frame
	 * @param frame
	 */
	private void marshallParametersFromGPU(HashMap<String, Object> frame) {
		int listCount = 0;
		for (int i = 0 ; i < devicePointers.size() ; i++) {
			Map<String, Type> environment = writer.getEnvironment();
			Type type = environment.get(parameters.get(i-listCount)); //compensating for offset of 1
			//convert from int* to [int]
			if (type instanceof Type.List) {
				marshallListFromGPU(frame,i-listCount,type);
				i++;
				listCount++;
			}else if (type instanceof Type.Int) {
				marshallIntFromGPU(frame,i-listCount);
			}else {
				InternalFailure.internalFailure("Could not unmarshall " +
						"unrecognised type", writer.getPtxFile().getPath() , type);
			}
		}
	}
	/**
	 * Marshalls a list from the GPU and places it on the frame.
	 * @param frame
	 * @param index
	 * @param type
	 */
	private void marshallListFromGPU(HashMap<String, Object> frame, int index,Type type) {
		String name = parameters.get(index);
		ArrayList<Integer> listObject = (ArrayList<Integer>) frame.get(name);
		//prepare variables for copying
		int length = listObject.size();
		int data[] = new int[length];
		cuMemcpyDtoH(Pointer.to(data), devicePointers.get(index), length*Sizeof.INT);
		ArrayList<Integer> newlist = new ArrayList<Integer>();
		//copy data to new list
		for (int i = 0 ; i < data.length ; i++) {
			newlist.add(data[i]);
		}
		//change frame to new value
		frame.put(name, newlist);
	}
	/**
	 * Marshalls a single int from the GPU and places it on the frame.
	 * @param frame
	 * @param index
	 */
	private void marshallIntFromGPU(HashMap<String, Object> frame, int index) {
		String name = parameters.get(index);
		int[] data = new int[1];
		//copy only one integer (maybe 4 bytes)
		cuMemcpyDtoH(Pointer.to(data), devicePointers.get(index), Sizeof.INT);
		//change frame
		frame.put(name, data[0]);
	}

	/**
	 * Marshalls the data from the parameters into an appropriate format and
	 * allocates memory appropriately, filling in pointers for kernel parameters.
	 * This method is called to upload data onto the GPU.
	 * @param frame
	 * @return
	 *
	 * @invariant symbolTypes.size() == params.size()
	 */
	private List<CUdeviceptr> marshallParametersToGPU(HashMap<String, Object> frame) {
		for (int i = 0 ; i < parameters.size() ; i++) {
			Type type = writer.getEnvironment().get(parameters.get(i));
			if (type instanceof Type.List) {
				marshallListToGPU(frame, i , (Type.List) type);
			}else if (type instanceof Type.Int) {
				marshallIntToGPU(frame, i);
			}else {
				InternalFailure.internalFailure("Could not marshall paramater to GPU",
						file.getName(), type);
			}
		}
		return devicePointers;
	}
	/**
	 * Marshall a single integer into the <i>index</i>th parameter
	 * @param frame
	 * @param index
	 */
	private void marshallIntToGPU(HashMap<String, Object> frame, int index) {
		int value = (Integer) frame.get(parameters.get(index));
		CUdeviceptr dpointer = generateH2DPointer(1,
				new int[] {value});
		devicePointers.add(dpointer);
	}
	/**
	 *
	 * @param frame
	 * @param index
	 * @param type
	 */
	private void marshallListToGPU(HashMap<String, Object> frame, int index,
			Type.List type) {
		int expectedLength;
		// then the next argument has to be the list length
		if (type.getElement() instanceof Type.Int) {
			String name = parameters.get(index);
			ArrayList<Integer> listObject = (ArrayList<Integer>) frame
					.get(name);
			expectedLength = listObject.size();
			int[] array = new int[expectedLength];
			// unrap all values in array to int type
			for (int j = 0; j < expectedLength; j++)
				array[j] = listObject.get(j);
			CUdeviceptr dpointer = generateH2DPointer(expectedLength,
					array);
			devicePointers.add(dpointer);
			//now compute the length argument and add it to the deviceptr list
			int[] lengthValue = new int []{expectedLength};
			CUdeviceptr lengthPtr = new CUdeviceptr();
			cuMemAlloc(lengthPtr, Sizeof.INT);
			cuMemcpyHtoD(lengthPtr, Pointer.to(lengthValue), Sizeof.INT);
			devicePointers.add(lengthPtr);

		}else {
			InternalFailure.internalFailure("Can only allocate pointer " +
					"for flat list of element type int" + index, writer.getPtxFile().getPath(), type);
		}
	}

	/**
	 * Allocates device memory for the integer array and returns a pointer to
	 * it.
	 *
	 * @param expectedLength
	 * @param intArray
	 * @return
	 */
	private CUdeviceptr generateH2DPointer(int expectedLength, int[] intArray) {
		CUdeviceptr dpointer = new CUdeviceptr();
		cuMemAlloc(dpointer, expectedLength * Sizeof.INT);
		cuMemcpyHtoD(dpointer, Pointer.to(intArray), expectedLength
				* Sizeof.INT);
		return dpointer;
	}

}
