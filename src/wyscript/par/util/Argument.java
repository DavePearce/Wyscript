package wyscript.par.util;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;

import java.util.ArrayList;
import java.util.Map;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

import wyscript.lang.Type;

public abstract class Argument {
	public abstract void write(Map<String,Object> frame , CUdeviceptr ptr);
	public abstract void read(Map<String,Object> frame , CUdeviceptr ptr);
	public abstract String getCType();
	public final String name;
	protected boolean hasAllocated = false;
	@Override
	public String toString() {
		return "Argument [name=" + name + "]";
	}

	public Argument(String name) {
		this.name = name;
	}
	/**
	 * Wraps the length of a flat list.
	 * @author antunomate
	 *
	 */
	public static class Length1D extends Argument {
		public Length1D(String name) {
			super(name);
		}
		@Override
		public void write(Map<String, Object> env, CUdeviceptr ptr) {
			ArrayList<?> list = (ArrayList<?>) env.get(name);
			int[] size = new int []{list.size()};
			if (!hasAllocated) {
				cuMemAlloc(ptr, Sizeof.INT);
				hasAllocated = true;
			}
			cuMemcpyHtoD(ptr, Pointer.to(size), Sizeof.INT);
		}
		@Override
		public void read(Map<String, Object> frame, CUdeviceptr ptr) {
			//does nothing
		}
		@Override
		public String getCType() {
			return "int*";
		}
	}
	/**
	 * Wraps the height or width of a flat list.
	 * A flag in the constructor is used to set whether this argument
	 * is for either width or height.
	 * @author antunomate
	 *
	 */
	public static class Length2D extends Argument {
		public final boolean isHeight;
		public Length2D(String name , boolean isHeight) {
			super(name);
			this.isHeight = isHeight;
		}
		@Override
		public String toString() {
			return "Length2D [name = "+name+", isHeight=" + isHeight + "]";
		}
		@Override
		public void write(Map<String, Object> env, CUdeviceptr ptr) {
			ArrayList<?> list = (ArrayList<?>) env.get(name);
			int rows = list.size();
			if (isHeight) {
				int[] value = new int[] { rows };
				if (!hasAllocated) {
					cuMemAlloc(ptr, Sizeof.INT);
					hasAllocated = true;
				}
				cuMemcpyHtoD(ptr, Pointer.to(value), Sizeof.INT);
			}else {
				//scan every row and get the greatest height
				int width = -1;
				for (int i = 0;i<rows;i++) {
					int w = ((ArrayList<?>)list.get(i)).size();
					if (w>width) {
						width = w;
					}
				}
				int[] value = new int[] { width };
				if (!hasAllocated) {
					cuMemAlloc(ptr, Sizeof.INT);
					hasAllocated = true;
				}
				cuMemcpyHtoD(ptr, Pointer.to(value), Sizeof.INT);
			}
		}
		@Override
		public void read(Map<String, Object> frame, CUdeviceptr ptr) {
			// this method will only read in those data values for which
			//there is space available

		}
		@Override
		public String getCType() {
			return "int*";
		}
	}
	public static class SingleInt extends Argument {
		public SingleInt(String name) {
			super(name);
		}
		@Override
		public void write(Map<String, Object> env, CUdeviceptr ptr) {
			int[] value = new int[] { (Integer) env.get(name) };
			if (!hasAllocated) {
				cuMemAlloc(ptr, Sizeof.INT);
				hasAllocated = true;
			}
			cuMemcpyHtoD(ptr, Pointer.to(value), Sizeof.INT);
		}
		@Override
		public void read(Map<String, Object> frame, CUdeviceptr ptr) {
			int[] value = new int []{0};
			cuMemcpyDtoH(Pointer.to(value),ptr, Sizeof.INT);
			frame.put(name, value[0]);
		}
		@Override
		public String getCType() {
			return "int*";
		}

	}
	public static class List1D extends Argument {
		int cachedSize;
		public List1D(String name) {
			super(name);
		}
		@Override
		public void write(Map<String, Object> env, CUdeviceptr ptr) {
			ArrayList<Integer> list = (ArrayList<Integer>) env.get(name);
			int length = list.size();
			if (length != cachedSize) {
				cachedSize = length;
				hasAllocated = false;
			}
			int[] values = new int[length];
			for (int i = 0; i < length ; i++) {
				values[i] = list.get(i);
			}
			if (!hasAllocated) {
				cuMemAlloc(ptr, cachedSize*Sizeof.INT);
				hasAllocated = true;
			}
			cuMemcpyHtoD(ptr, Pointer.to(values), length*Sizeof.INT);
		}
		@Override
		public void read(Map<String, Object> frame, CUdeviceptr ptr) {
			ArrayList<Integer> list = (ArrayList<Integer>) frame.get(name);
			ArrayList<Integer> newlist = new ArrayList<Integer>();
			int[] value = new int [list.size()];
			cuMemcpyDtoH(Pointer.to(value),ptr, Sizeof.INT*list.size());
			for (int v : value) {
				newlist.add(v);
			}
			frame.put(name, newlist);
		}
		@Override
		public String getCType() {
			return "int*";
		}

	}
	public static class List2D extends Argument {
		int height;
		int width;
		public List2D(String name) {
			super(name);
		}
		@Override
		public void write(Map<String, Object> env, CUdeviceptr ptr) {
			ArrayList<?> listOfLists = (ArrayList<?>) env.get(name);
			height = listOfLists.size();
			width = -1;
			for (int y = 0 ; y < height ; y++) {
				ArrayList list = (ArrayList) listOfLists.get(0);
				if (list.size()>width) {
					width = list.size();
				}
			}
			int[] array = new int[width*height];
			for (int y = 0 ; y < height ; y++) {
				ArrayList list = (ArrayList) listOfLists.get(y);
				for (int x = 0 ; x < list.size() ; x++) {
					array[y*width + x] = (Integer) list.get(x);
				}
			}
			//data copied!
			if (!hasAllocated) {
				cuMemAlloc(ptr, width*height*Sizeof.INT);
				hasAllocated = true;
			}
			cuMemcpyHtoD(ptr, Pointer.to(array), width*height*Sizeof.INT);
		}
		@Override
		public void read(Map<String, Object> frame, CUdeviceptr ptr) {
			int[] array = new int[width*height];
			cuMemcpyDtoH(Pointer.to(array),ptr, width*height*Sizeof.INT);
			ArrayList<?> listOfLists = (ArrayList<?>) frame.get(name);
			for (int y = 0 ; y < height ; y++) {
				ArrayList list = (ArrayList) listOfLists.get(y);
				for (int x = 0 ; x < list.size() ; x++) {
					list.set(x, array[y*width + x]);
				}
			}
		}
		@Override
		public String getCType() {
			return "int*";
		}
	}
	public static class IndexOffset extends Argument {
		int length;
		public IndexOffset(String name , int length) {
			super(name);
			this.length  = length;
		}
		public void setOffset(int length) {
			this.length = length;
		}
		@Override
		public void write(Map<String, Object> frame, CUdeviceptr ptr) {
			int[] value = new int[] { length };
			if (!hasAllocated) {
				cuMemAlloc(ptr, Sizeof.INT);
				hasAllocated = true;
			}
			cuMemcpyHtoD(ptr, Pointer.to(value), Sizeof.INT);
		}

		@Override
		public void read(Map<String, Object> frame, CUdeviceptr ptr) {
			return;
		}

		@Override
		public String getCType() {
			return "int*";
		}
	}
}
