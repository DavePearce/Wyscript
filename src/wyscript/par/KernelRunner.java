package wyscript.par;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import jcuda.*;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;
import wyscript.lang.*;
import wyscript.util.SyntaxError.InternalFailure;

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
	List<Type> paramTypes;
	List<CUdeviceptr> devicePointers;

	File file;

	int gridDim;
	int blockDim;

	public KernelRunner(KernelWriter writer) {
		this.writer = writer;
		file = writer.getPtxFile();
	}

	/**
	 * This will initialise the kernel from the given file name
	 *
	 * @param ptxFileName
	 */
	public void initialise() {
		String funcname = writer.getFuncName();
		String ptxFileName = file.getAbsolutePath();
		// initialise driver and create context
		cuInit(0);
		CUdevice device = new CUdevice();
		cuDeviceGet(device, 0);
		CUcontext context = new CUcontext();
		cuCtxCreate(context, 0, device);
		//this.context = context;
		// Load the ptx file.
		CUmodule module = new CUmodule();
		cuModuleLoad(module, ptxFileName);
		// Obtain a function pointer to the correct function function.
		CUfunction function = new CUfunction();
		cuModuleGetFunction(function, module, funcname);
		this.function = function;
		// host input data ready to be filled
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
		initialise();
		//paramPointers.add(Pointer.to(new int[] { numParams })); // add the count
		List<CUdeviceptr> pointers = marshallParametersToGPU(frame);
		Pointer parametersPointer = getPointerToParams(pointers);
		marshallParametersToGPU(frame);
		cuLaunchKernel(function, gridDim, 1, 1, blockDim, 1, 1, 0, null,
				parametersPointer, null);
		cuCtxSynchronize();
		marshallParametersFromGPU(frame);
		return null; //TODO check whether this should be changed
	}
	/**
	 * Returns a pointer to the parameters for the kernel
	 * @param devicePointers
	 * @return
	 */
	private Pointer getPointerToParams(List<CUdeviceptr> devicePointers) {
		int length = devicePointers.size();
		NativePointerObject[] devicePointerArray = new NativePointerObject[length+1];
		devicePointerArray[0] = Pointer.to(new int[]{length});
		for (int i = 1 ; i < devicePointerArray.length + 1 ; i++) {
			devicePointerArray[i] = Pointer.to(devicePointers.get(i));
		}
		//now take the array full of parameters and get a pointer to it
		Pointer toParams = Pointer.to(devicePointerArray);
		return toParams;
	}

	/**
	 * Responsible for converting data back from the GPU to the format
	 * expected on the frame
	 * @param frame
	 */
	private void marshallParametersFromGPU(HashMap<String, Object> frame) {
		for (int i = 0 ; i < devicePointers.size() ; i++) {
			Type type = paramTypes.get(i); //compensating for offset of 1
			//convert from int* to [int]
			if (type instanceof Type.List) {
				marshallListFromGPU(frame,i,type);
				i++;
			}else if (type instanceof Type.Int) {
				marshallIntFromGPU(frame,i);
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
			Type type = paramTypes.get(i);
			if (type instanceof Type.List) {
				marshallListToGPU(frame, i , (Type.List) type);
			}else if (type instanceof Type.Int) {
				marshallIntToGPU(frame, i);
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
		CUdeviceptr dpointer = generatePointer(1,
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
			CUdeviceptr dpointer = generatePointer(expectedLength,
					array);
			devicePointers.add(dpointer);
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
	private CUdeviceptr generatePointer(int expectedLength, int[] intArray) {
		CUdeviceptr dpointer = new CUdeviceptr();
		cuMemAlloc(dpointer, expectedLength * Sizeof.INT);
		cuMemcpyHtoD(dpointer, Pointer.to(intArray), expectedLength
				* Sizeof.INT);
		return dpointer;
	}

}
