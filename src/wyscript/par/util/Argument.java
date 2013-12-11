package wyscript.par.util;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import java.util.ArrayList;
import java.util.Map;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

import wyscript.lang.Type;

public abstract class Argument {
	public abstract void write(Map<String,Object> frame , CUdeviceptr ptr);
	public final String name;

	public Argument(String name) {
		this.name = name;
	}
	public class Length1D extends Argument {
		public Length1D(String name) {
			super(name);
		}
		@Override
		public void write(Map<String, Object> env, CUdeviceptr ptr) {
			ArrayList<?> list = (ArrayList<?>) env.get(name);
			int[] size = new int []{list.size()};
			cuMemAlloc(ptr, Sizeof.INT);
			cuMemcpyHtoD(ptr, Pointer.to(size), Sizeof.INT);
		}

	}
	public class SingleInt extends Argument {
		public SingleInt(String name) {
			super(name);
		}
		@Override
		public void write(Map<String, Object> env, CUdeviceptr ptr) {
			int[] value = new int[] { (Integer) env.get(name) };
			cuMemAlloc(ptr, Sizeof.INT);
			cuMemcpyHtoD(ptr, Pointer.to(value), Sizeof.INT);
		}

	}
	public class List1D extends Argument {
		public List1D(String name) {
			super(name);
		}
		@Override
		public void write(Map<String, Object> env, CUdeviceptr ptr) {
			ArrayList<Integer> list = (ArrayList<Integer>) env.get(name);
			int length = list.size();
			int[] values = new int[length];
			for (int i = 0; i < length ; i++) {
				values[i] = list.get(i);
			}
			cuMemAlloc(ptr, length*Sizeof.INT);
			cuMemcpyHtoD(ptr, Pointer.to(values), Sizeof.INT);
		}

	}
	public class List2D extends Argument {
		public List2D(String name) {
			super(name);
		}
		@Override
		public void write(Map<String, Object> env, CUdeviceptr ptr) {
			ArrayList<?> list = (ArrayList<?>) env.get(name);
			int rows = list.size();
			for (int y = 0 ; y < rows ; y++) {
				//TODO find out the type of 
			}
		}

	}
}
