package wyscript.testing;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Test;

import static wyscript.testing.TestUtils.*;
public class ParallelValidTests {
	private static final String testdir = "partests/runtests/";

	protected void runInterpreterTest(String name) {
		// The name of the file which contains the output for this test
		String sampleOutputFile = testdir + File.separatorChar + name
				+ ".sysout";
		// Classpath to project root
		String classPath = "../../bin";

		// First, execute the While program using the interpreter
		String output = TestUtils.exec(classPath, testdir, "wyscript.Main", name + ".wys");

		// Second, compare the output!
		TestUtils.compare(output, sampleOutputFile);
	}
	@Test
	public void parFor_Valid_1() {
		runInterpreterTest("ParFor_Valid_1");
	}
	@Test
	public void parFor_Valid_2() {
		runInterpreterTest("ParFor_Valid_2");
	}
	@Test
	public void parFor_Valid_3() {
		runInterpreterTest("ParFor_Valid_3");
	}
	@Test
	public void parFor_Valid_4() {
		runInterpreterTest("ParFor_Valid_4");
	}

}
