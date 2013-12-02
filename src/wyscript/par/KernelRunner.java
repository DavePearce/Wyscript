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

	// List<CUdeviceptr> devicePointers = new ArrayList<CUdeviceptr>();
	Pointer[] kernelParameters;
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
		paramPointers.add(Pointer.to(new int[] { numParams })); // add the count
																// argument
		// sanity check the input value size
		boolean expectingLengthArg = false;
		int expectedLength = -1;
		for (int i = 0; i < symbolTypes.size(); i++) {
			Type type = symbolTypes.get(i);
			if (expectingLengthArg) {
				// String name = params.get(i); //don't actually need to get
				// name
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
				// then the next argument has to be the list length
				if (((Type.List) type).getElement() instanceof Type.Int) {
					String name = params.get(i);
					ArrayList<Integer> listObject = (ArrayList<Integer>) frame
							.get(name);
					expectedLength = listObject.size();
					int[] array = new int[expectedLength];
					// copy values over, ensuring that they are unwrapped form
					// Integer type.
					for (int j = 0; j < expectedLength; j++)
						array[j] = listObject.get(j);
					CUdeviceptr dpointer = generatePointer(expectedLength,
							array);
					kernelParameters[i + 1] = Pointer.to(dpointer);
					// TODO I believe the value of the list (on the frame) is
					// ArrayList<Object>
				}
			}// end processing list type
		}// end of the loop over symbol types
			// verify and check twice all the assumptions made up until this
			// point!
		Pointer parametersPointer = Pointer.to(kernelParameters);
		cuLaunchKernel(function, gridDim, 1, 1, blockDim, 1, 1, 0, null,
				parametersPointer, null);
		cuCtxSynchronize();
		return null; //TODO change me
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
