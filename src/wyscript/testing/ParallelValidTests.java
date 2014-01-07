package wyscript.testing;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Test;

import static wyscript.testing.TestUtils.*;
public class ParallelValidTests {
	private static final String testdir = "partests/runtests/";
	private static final String JCUDA_HOME = "../../cudalib/jcuda-0.5.0b.jar";

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
	@Test
	public void parFor_valid_15() {
		runParInterpreterTest("ParFor_Valid_15");
	}
	@Test
	public void parFor_valid_16() {
		runParInterpreterTest("ParFor_Valid_16");
	}
	@Test
	public void parFor_valid_17() {
		runParInterpreterTest("ParFor_Valid_17");
	}
	@Test
	public void parFor_valid_18() {
		runParInterpreterTest("ParFor_Valid_18");
	}
	@Test
	public void parFor_valid_19() {
		runParInterpreterTest("ParFor_Valid_19");
	}
	@Test
	public void parFor_valid_20() {
		runParInterpreterTest("ParFor_Valid_20");
	}
	@Test
	public void parFor_valid_21() {
		runParInterpreterTest("ParFor_Valid_21");
	}
	@Test
	public void parFor_valid_22() {
		runParInterpreterTest("ParFor_Valid_22");
	}
	@Test
	public void parFor_valid_23() {
		runParInterpreterTest("ParFor_Valid_23");
	}
	@Test
	public void parFor_valid_24() {
		runParInterpreterTest("ParFor_Valid_24");
	}
	@Test
	public void parFor_valid_25() {
		runParInterpreterTest("ParFor_Valid_25");
	}
	@Test
	public void parFor_valid_26() {
		runParInterpreterTest("ParFor_Valid_26");
	}
	@Test
	public void parFor_valid_27() {
		runParInterpreterTest("ParFor_Valid_27");
	}
	@Test
	public void parFor_real_1() {
		runParInterpreterTest("ParFor_Real_1");
	}
}
