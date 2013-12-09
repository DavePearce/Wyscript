package wyscript.testing;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Test;

import static wyscript.testing.TestUtils.*;
public class ParallelValidTests {
	private static final String testdir = "partests/runtests/";
	private static final String JCUDA_HOME = "";

	protected void runParInterpreterTest(String name) {
		// The name of the file which contains the output for this test
		String sampleOutputFile = testdir + File.separatorChar + name
				+ ".sysout";
		// Classpath to project root
		String classPath = "../../bin/:"+JCUDA_HOME;

		// First, execute the While program using the interpreter
		String output = TestUtils.parExec(classPath, testdir, "wyscript.Main", name + ".wys");

		// Second, compare the output!
		TestUtils.compare(output, sampleOutputFile);
	}
	@Test
	public void parFor_Valid_1() {
		runParInterpreterTest("ParFor_Valid_1");
	}
	@Test
	public void parFor_Valid_2() {
		runParInterpreterTest("ParFor_Valid_2");
	}
	@Test
	public void parFor_Valid_3() {
		runParInterpreterTest("ParFor_Valid_3");
	}
	@Test
	public void parFor_Valid_4() {
		runParInterpreterTest("ParFor_Valid_4");
	}
	@Test
	public void parFor_Valid_5() {
		runParInterpreterTest("ParFor_Valid_5");
	}
	@Test
	public void parFor_Valid_6() {
		runParInterpreterTest("ParFor_Valid_6");
	}
	@Test
	public void parFor_valid_7() {
		runParInterpreterTest("ParFor_Valid_7");
	}
	@Test
	public void parFor_valid_8() {
		runParInterpreterTest("ParFor_Valid_8");
	}
	@Test
	public void parFor_valid_9() {
		runParInterpreterTest("ParFor_Valid_9");
	}
	@Test
	public void parFor_valid_10() {
		runParInterpreterTest("ParFor_Valid_10");
	}
	@Test
	public void parFor_valid_11() {
		runParInterpreterTest("ParFor_Valid_11");
	}
	@Test
	public void parFor_valid_12() {
		runParInterpreterTest("ParFor_Valid_12");
	}
	@Test
	public void parFor_valid_13() {
		runParInterpreterTest("ParFor_Valid_13");
	}
	@Test
	public void parFor_valid_14() {
		runParInterpreterTest("ParFor_Valid_14");
	}
}
