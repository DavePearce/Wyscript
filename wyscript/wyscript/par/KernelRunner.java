package wyscript.par;

import jcuda.*;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;
import jcuda.driver.JCudaDriver.*;

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
	private
	//List
	//private
	/**
	 * This will initialise the kernel from the given file name
	 * @param ptxFileName
	 */
	public void initialise(String ptxFileName) {
		//initialise driver and create context
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
        this.context = context;
        //now get function pointer
	}
	/**
	 * Calling run uploads the kernel to the GPU
	 */
	public void run() {
		//this will actually upload the function onto the GPU
		//as well as the data and run it
		//wait, then copy back the result and marshall it appropriately
	}
}
