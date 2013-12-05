package wyscript.testing;

import java.util.HashMap;
import java.util.Map;
import org.junit.Test;

import wyscript.Interpreter;
import wyscript.lang.Type;
import wyscript.lang.WyscriptFile;
import wyscript.par.KernelGenerator;
import wyscript.util.TypeChecker;
import static wyscript.testing.TestUtils.*;

public class KernelTests {
	private static final String testDir = "../Wyscript/partests/";

	@Test
	public void test_write() {
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("i",new Type.Int());
		env.put("l",new Type.List(new Type.Int()));
		writerTest("test1",env);
	}
	@Test
	public void test2_write() {
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("i",new Type.Int());
		env.put("a",new Type.List(new Type.Int()));
		env.put("b",new Type.List(new Type.Int()));
		env.put("c",new Type.List(new Type.Int()));
		writerTest("test2",env);
	}
	@Test
	public void test3_write() {
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("i",new Type.Int());
		env.put("l",new Type.List(new Type.Int()));
		writerTest("test3",env);
	}
	@Test
	public void test4_write() {
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("i",new Type.Int());
		env.put("a",new Type.List(new Type.Int()));
		env.put("b",new Type.List(new Type.Int()));
		env.put("c",new Type.List(new Type.Int()));
		writerTest("test4",env);
	}
	@Test
	public void test5_write() {
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("i",new Type.Int());
		env.put("x",new Type.Int());
		env.put("c",new Type.List(new Type.Int()));
		writerTest("test5",env);
	}
	@Test
	public void test6_write() {
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("i",new Type.Int());
		env.put("a",new Type.List(new Type.Int()));
		env.put("b",new Type.List(new Type.Int()));
		env.put("c",new Type.List(new Type.Int()));
		env.put("x",new Type.List(new Type.Int()));
		writerTest("test6");
	}
	@Test
	public void test1_run() {
		WyscriptFile ast = parseForFile(testDir+"runtest1.wys");
		runRunTest(ast);
	}

	@Test
	public void test2_run() {
		WyscriptFile ast = parseForFile(testDir+"runtest2.wys");
		runRunTest(ast);
	}
	@Test
	public void test3_run() {
		WyscriptFile ast = parseForFile(testDir+"runtest3.wys");
		runRunTest(ast);
	}
	@Test
	public void test4_run() {
		WyscriptFile ast = parseForFile(testDir+"runtest4_gpu.wys");
		runRunTest(ast);
	}
	private void runRunTest(WyscriptFile ast) {
		TypeChecker checker = new TypeChecker();
		checker.check(ast);
		KernelGenerator.generateKernels(ast);
		Interpreter interpreter = new Interpreter();
		long time = System.currentTimeMillis();
		interpreter.run(ast);
		System.out.println("Took: "+(System.currentTimeMillis()-time));
	}
}
