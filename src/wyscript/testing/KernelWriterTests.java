package wyscript.testing;

import java.util.HashMap;
import java.util.Map;
import org.junit.Test;

import wyscript.lang.Type;
import static wyscript.testing.TestUtils.*;

public class KernelWriterTests {
	private static final String testDir = "/am/state-opera/home1/antunomate/summer_research/wy_material/WyScript_fork/Wyscript/partests/";

	@Test
	public void test() {
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("i",new Type.Int());
		env.put("l",new Type.List(new Type.Int()));
		performTest("test1",env);
	}
	@Test
	public void test2() {
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("i",new Type.Int());
		env.put("a",new Type.List(new Type.Int()));
		env.put("b",new Type.List(new Type.Int()));
		env.put("c",new Type.List(new Type.Int()));
		performTest("test2",env);
	}
	@Test
	public void test3() {
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("i",new Type.Int());
		env.put("l",new Type.List(new Type.Int()));
		performTest("test3",env);
	}
	@Test
	public void test4() {
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("i",new Type.Int());
		env.put("a",new Type.List(new Type.Int()));
		env.put("b",new Type.List(new Type.Int()));
		env.put("c",new Type.List(new Type.Int()));
		performTest("test4",env);
	}
	@Test
	public void test5() {
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("i",new Type.Int());
		env.put("x",new Type.Int());
		env.put("c",new Type.List(new Type.Int()));
		performTest("test5",env);
	}
	@Test
	public void test6() {
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("i",new Type.Int());
		env.put("a",new Type.List(new Type.Int()));
		env.put("b",new Type.List(new Type.Int()));
		env.put("c",new Type.List(new Type.Int()));
		performTest("test6",env);
	}
}
