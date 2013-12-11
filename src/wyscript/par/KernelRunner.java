package wyscript.par;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import jcuda.*;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;
import wyscript.lang.*;
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
	//Three lists used to track the name, type and device pointer of elements
	List<String> parameters;
	List<CUdeviceptr> devicePointers;

	File file;

	int gridDim = 16;
	int blockDim = 256;
	private CUcontext context;
	private LoopModule module;

	public KernelRunner(LoopModule module) {
		this.module = module;
		file = module.getPtxFile();
		parameters = module.getParameters();
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
		//now initialise the entire thing
	}
	/**
	 * Helper method to stop program if there is a cuda-related error
	 * @param result The cuda error code (or success code)
	 */
	private void stopIfFailed(int result) {
		if (result != CUDA_SUCCESS) {
			InternalFailure.internalFailure(
					"Failure with code"+result+". Got error: "+
			cudaGetErrorString(result), file.getName(), module.getLoop());
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
		int result = cuLaunchKernel(function,
				gridDim, 1, 1,
				blockDim, 1, 1,
				0, null,
				Pointer.to(parametersPointer), null);
		int syncResult = cuCtxSynchronize();
		stopIfFailed(syncResult);
		if (result != CUDA_SUCCESS) {
			InternalFailure.internalFailure("Kernel did not launch successfully." +
					cudaGetErrorString(result), file.getName(), module.getLoop());
		}
		marshalParametersFromGPU(frame);
		cuCtxDestroy(context);
		cleanUp(pointers);
		return null; //no need to return any particular object
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
	private void marshalParametersFromGPU(HashMap<String, Object> frame) {
		int listCount = 0;
		for (int i = 0 ; i < devicePointers.size() ; i++) {
			Map<String, Type> environment = module.getEnvironment();
			String parameter = parameters.get(i-listCount);
			Type type = environment.get(parameter); //compensating for offset of 1
			//convert from int* to [int]
			if (type instanceof Type.List) {
				marshalFromGPUList(frame,parameter,i,type);
				//next instructions skip the length argument
				i++;
				listCount++;
			}else if (type instanceof Type.Int) {
				marshalFromGPUInt(frame,parameter,i);
			}else {
				InternalFailure.internalFailure("Could not unmarshall " +
						"unrecognised type", module.getPtxFile().getPath() , type);
			}
		}
	}
	/**
	 * Marshalls a list from the GPU and places it on the frame.
	 * @param frame
	 * @param index
	 * @param type
	 */
	private void marshalFromGPUList(HashMap<String, Object> frame, String name , int index,Type type) {
		ArrayList<Integer> listObject;
		try {
			listObject = (ArrayList<Integer>) frame.get(name);
		}catch (ClassCastException e) {
			InternalFailure.internalFailure("Runtime type error. Expected '"
		+name+"' to be type java.util.ArrayList. Was type "+e.getClass(), file.getPath(), type);
			return;
		}
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
	 * @param name
	 * @param paramIndex
	 */
	private void marshalFromGPUInt(HashMap<String, Object> frame, String name, int paramIndex) {
		int[] data = new int[1];
		//copy only one integer (maybe 4 bytes)
		cuMemcpyDtoH(Pointer.to(data), devicePointers.get(paramIndex), Sizeof.INT);
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
	 */
	private List<CUdeviceptr> marshalParametersToGPU(HashMap<String, Object> frame) {
		for (int i = 0 ; i < parameters.size() ; i++) {
			Type type = module.getEnvironment().get(parameters.get(i));
			if (type instanceof Type.List) {
				if (((Type.List) type).getElement() instanceof Type.Int) {
					String name = parameters.get(i);
					marshalFlatListToGPU(frame, name);
				}else if (((Type.List) type).getElement() instanceof Type.List) {
					String name = parameters.get(i);
					marshal2DListToGPU(frame, name);
				}
					else {
					InternalFailure.internalFailure("Can only allocate pointer " +
							"for flat list of element type int", file.getPath(), type);
				}
			}else if (type instanceof Type.Int) {
				int value = (Integer) frame.get(parameters.get(i));
				marshalToGPUInt(frame, value);
			}else {
				InternalFailure.internalFailure("Could not marshall paramater to GPU",
						file.getName(), type);
			}
		}
		return devicePointers;
	}
	private void marshal2DListToGPU(HashMap<String, Object> frame, String name) {
		// then the next argument has to be the list length
		int height;
		int width = -1;
		ArrayList<?> listObject = (ArrayList<?>) frame.get(name);
		height = listObject.size();
		for (Object element : listObject) {
			ArrayList<Integer> list = (ArrayList<Integer>) element;

		}
		int expectedLength = listObject.size();
		int[] array = new int[width*height];
		// unrap all values in array to int type
		for (int j = 0; j < expectedLength; j++) array[j] = (Integer) listObject.get(j);
		CUdeviceptr dpointer = generateH2DPointer(array);
		devicePointers.add(dpointer);
		//now compute the length argument and add it to the deviceptr list
		int[] lengthValue = new int []{expectedLength};
		CUdeviceptr lengthPtr = new CUdeviceptr();
		cuMemAlloc(lengthPtr, Sizeof.INT);
		cuMemcpyHtoD(lengthPtr, Pointer.to(lengthValue), Sizeof.INT);
		devicePointers.add(lengthPtr);
	}

	/**
	 * Marshall a single integer into the <i>index</i>th parameter
	 * @param frame
	 * @param index
	 */
	private void marshalToGPUInt(HashMap<String, Object> frame, int value) {
		CUdeviceptr dpointer = generateH2DPointer(new int[] {value});
		devicePointers.add(dpointer);
	}
	/**
	 *
	 * @param frame
	 * @param index
	 * @param type
	 */
	private void marshalFlatListToGPU(HashMap<String, Object> frame, String name) {
		int expectedLength;
		// then the next argument has to be the list length
		ArrayList<Integer> listObject = (ArrayList<Integer>) frame.get(name);
		expectedLength = listObject.size();
		int[] array = new int[expectedLength];
		// unrap all values in array to int type
		for (int j = 0; j < expectedLength; j++) array[j] = listObject.get(j);
		CUdeviceptr dpointer = generateH2DPointer(array);
		devicePointers.add(dpointer);
		//now compute the length argument and add it to the deviceptr list
		int[] lengthValue = new int []{expectedLength};
		CUdeviceptr lengthPtr = new CUdeviceptr();
		cuMemAlloc(lengthPtr, Sizeof.INT);
		cuMemcpyHtoD(lengthPtr, Pointer.to(lengthValue), Sizeof.INT);
		devicePointers.add(lengthPtr);
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
