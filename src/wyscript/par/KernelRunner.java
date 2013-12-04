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
	/*
	 * I guess the required elements to the kernel runner include a list of
	 * parameters to the function, and the function pointer of course.
	 */
	private CUcontext context;
	private CUfunction function;
	List<String> params;

	List<Type> symbolTypes;

	// List<CUdeviceptr> devicePointers = new ArrayList<CUdeviceptr>();
	Pointer[] kernelParameters;
	List<CUdeviceptr> devicePointers;
	private int numParams; // TODO ensure task parameter is filled with correct
							// value
	private KernelWriter writer;

	int gridDim;
	int blockDim;

	public KernelRunner(KernelWriter writer) {
		this.writer = writer;
	}

	/**
	 * This will initialise the kernel from the given file name
	 *
	 * @param ptxFileName
	 */
	public void initialise(File ptxFile, String funcName) {
		String ptxFileName = ptxFile.getAbsolutePath();
		// initialise driver and create context
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
		// sanity check the number of arguments -- values is the number of input
		// objects
		List<NativePointerObject> paramPointers = new ArrayList<NativePointerObject>();
		paramPointers.add(Pointer.to(new int[] { numParams })); // add the count														// argument
		// sanity check the input value size
		marshallParametersIn(frame);
		Pointer parametersPointer = Pointer.to(kernelParameters);
		cuLaunchKernel(function, gridDim, 1, 1, blockDim, 1, 1, 0, null,
				parametersPointer, null);
		cuCtxSynchronize();
		marshallParametersOut(frame);
		return null; //TODO change me
	}
	/**
	 * Responsible for converting data back from the GPU to the format
	 * expected on the frame
	 * @param frame
	 */
	private void marshallParametersOut(HashMap<String, Object> frame) {
		//TODO SERIOUS PROBLEM : how to deal with the length arguments in parameter list
		for (int i = 0 ; i < kernelParameters.length ; i++) {
			Type type = symbolTypes.get(i); //compensating for offset of 1
			String name = params.get(i);
			int size;
			//convert from int* to [int]
			if (type instanceof Type.List && ((Type.List)type).getElement()
					instanceof Type.Int) {
				//Type.List list = frame.get(name);
				ArrayList<Object>  vallist = (ArrayList<Object>) frame.get(name);
				size = vallist.size();
				int[] newVals = new int[size];
				//fills the newvals array with the values from the device.
				cuMemcpyDtoH(Pointer.to(newVals), devicePointers.get(i), size *
						Sizeof.INT);
				ArrayList<Object> newList = new ArrayList<Object>();
				//copy the values over
				for (int j = 0 ; j < size ; j++) newList.add(newVals[j]);
				//now object reassigned
				frame.put(name,newList);
			}else if (type instanceof Type.Int) {

			}else {
				InternalFailure.internalFailure("Could not unmarshall " +
						"unrecognised type", writer.getPtxFile().getPath() , type);
			}
		}
	}

	/**
	 * Marshalls the data from the parameters into an appropriate format and
	 * allocates memory appropriately, filling in pointers for kenrel parameters.
	 * This method is called to upload data onto the GPU
	 * @param frame
	 */
	private void marshallParametersIn(HashMap<String, Object> frame) {
		boolean expectingLengthArg = false;
		int expectedLength = -1;
		for (int i = 0; i < symbolTypes.size(); i++) {
			Type type = symbolTypes.get(i);
			if (expectingLengthArg) {
				//then the pointer for the length argument is allocated
				Type symType = symbolTypes.get(i);
				if (symType instanceof Type.Int) {
					// paramPointers.add(Pointer.to(); //add the count argument
				} else {
					InternalFailure.internalFailure("Was expecting length "
							+ "argument at parameter " + i, writer.getPtxFile()
							.getPath(), type);
				}
				expectingLengthArg = false;
				expectedLength = -1;
			}// end processing int type
			else if (type instanceof Type.List) {
				marshallList(frame, i, type);
			}else if (type instanceof Type.Int) {
				marshallInt(frame, i);
			}else {

			}
		}
		//TODO investigate the behaviour of program at this point.
	}

	private void marshallInt(HashMap<String, Object> frame, int index) {
		int value = (Integer) frame.get(params.get(index));
		CUdeviceptr dpointer = generatePointer(1,
				new int[] {value});
		devicePointers.add(dpointer);
		kernelParameters[index+1] = Pointer.to(dpointer);
	}

	private void marshallList(HashMap<String, Object> frame, int index, Type type) {
		int expectedLength;
		// then the next argument has to be the list length
		if (((Type.List) type).getElement() instanceof Type.Int) {
			String name = params.get(index);
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
			kernelParameters[index + 1] = Pointer.to(dpointer);
			// TODO I believe the value of the list (on the frame) is
			// ArrayList<Object>
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
