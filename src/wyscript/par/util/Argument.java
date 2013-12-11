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
	public abstract void read(Map<String,Object> frame , CUdeviceptr ptr);
	public final String name;

	public Argument(String name) {
		this.name = name;
	}
	public static class Length1D extends Argument {
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
		@Override
		public void read(Map<String, Object> frame, CUdeviceptr ptr) {
			// TODO Auto-generated method stub

		}
	}
	public static class Length2D extends Argument {
		private boolean isHeight;
		public Length2D(String name , boolean isHeight) {
			super(name);
			this.isHeight = isHeight;
		}
		@Override
		public void write(Map<String, Object> env, CUdeviceptr ptr) {
			ArrayList<?> list = (ArrayList<?>) env.get(name);
			int rows = list.size();
			for (int y = 0 ; y < rows ; y++) {
				//TODO find out the type of the element in ArrayList (is it also arraylist or Expr.List
			}
		}
		@Override
		public void read(Map<String, Object> frame, CUdeviceptr ptr) {
			// TODO Auto-generated method stub

		}
	}
	public static class SingleInt extends Argument {
		public SingleInt(String name) {
			super(name);
		}
		@Override
		public void write(Map<String, Object> env, CUdeviceptr ptr) {
			int[] value = new int[] { (Integer) env.get(name) };
			cuMemAlloc(ptr, Sizeof.INT);
			cuMemcpyHtoD(ptr, Pointer.to(value), Sizeof.INT);
		}
		@Override
		public void read(Map<String, Object> frame, CUdeviceptr ptr) {
			// TODO Auto-generated method stub

		}

	}
	public static class List1D extends Argument {
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
		@Override
		public void read(Map<String, Object> frame, CUdeviceptr ptr) {
			// TODO Auto-generated method stub

		}

	}
	public static class List2D extends Argument {
		public List2D(String name) {

			super(name);
		}
		@Override
		public void write(Map<String, Object> env, CUdeviceptr ptr) {
			ArrayList<?> list = (ArrayList<?>) env.get(name);
			int rows = list.size();
			for (int y = 0 ; y < rows ; y++) {
				//TODO find out the type of the element in ArrayList (is it also arraylist or Expr.List
			}
		}
		@Override
		public void read(Map<String, Object> frame, CUdeviceptr ptr) {
			// TODO Auto-generated method stub

		}
	}
	public static Argument convertToArg(String name , Type type) {
		if (type instanceof Type.Int) {
			//simply return a single-int argument
			Argument arg = new SingleInt(name);
			return arg;
		}else if (type instanceof Type.List) {
			//differentiate between 1D and 2D lists
			Type elementType = (((Type.List) type).getElement());
			if (elementType instanceof Type.Int) {
				return new Argument.List1D(name);
			}else if (elementType instanceof Type.List) {
				if (((Type.List) elementType).getElement() instanceof Type.Int) {
					return new Argument.List2D(name);
				}
			}else {
				throw new IllegalArgumentException("Unknown type cannot be converted to kernel argument");
			}
		}
		//TODO implement the rest of me
		else throw new IllegalArgumentException("Unknown type cannot be converted to kernel argument");
		return null; //unreachable code
	}
}
