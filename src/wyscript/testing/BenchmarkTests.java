package wyscript.testing;

import static org.junit.Assert.fail;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Reader;

import org.junit.*;
import org.mozilla.javascript.Context;
import org.mozilla.javascript.Scriptable;
import org.mozilla.javascript.ScriptableObject;

public class BenchmarkTests {

	/**
	 * Path to test directory.
	 */
	private static String testdir = "benchmarks/";

	// ======================================================================
	// Test Harness
	// ======================================================================

	private void runTest(String name) {
		// The name of the file which contains the output for this test
		String generatedJavaScriptFile = testdir + File.separatorChar + name
				+ ".js";
		// The name of the file which contains the output for this test
		String sampleOutputFile = testdir + File.separatorChar + name
				+ ".sysout";

		// Classpath to project root
		String classPath = "../src";

		// First, we need to compile the given test into javascript
		String errors = TestUtils.exec(classPath, testdir, "wyscript.Main", "-js", name + ".wys");

		if(!errors.equals("")) {
			System.err.println(errors);
			fail(errors);
		}

		// Second, execute the generated JavaScript Program.
		String output = execJavaScript(generatedJavaScriptFile);

		// Third, compare the output!
		TestUtils.compare(output,sampleOutputFile);
	}

	/**
	 * Execute the main() method on a given (generated) Javascript file, and
	 * capture the output.
	 *
	 * @param filename Filename of generated JavaScript source file.
	 * @return
	 */
	private static String execJavaScript(String filename) {
		OutputStream out = new ByteArrayOutputStream();
	    try {
	      Reader file = new FileReader(new File(filename));
	      Context cxt = Context.enter();
	      Scriptable scope = cxt.initStandardObjects();


	      Object sysout = Context.javaToJS(new PrintStream(out), scope);
	      OutputStream err = new ByteArrayOutputStream();
	      Object syserr = Context.javaToJS(new PrintStream(err), scope);

	      ScriptableObject.putConstProperty(scope, "sysout", sysout);
	      ScriptableObject.putConstProperty(scope, "syserr", syserr);

	      //Set up the library
	      String lib = testdir + File.separatorChar + "Wyscript.js";
	      Reader library = new FileReader(new File(lib));
	      cxt.evaluateReader(scope, library, lib, 1, null);

	      cxt.evaluateReader(scope, file, filename, 1, null);
	      cxt.evaluateString(scope, "main()", "main", 1, null);

	      System.err.println(err);
	      return out.toString();
	    } catch (Exception ex) {
	      System.err.print(out);
	      ex.printStackTrace();
	      fail("Problem running compiled test");
	    } finally {
	      Context.exit();
	    }

	    return null;
	  }

	@Test
	public void Benchmark_Conways() {
		runTest("Benchmark_Conways");
	}

	@Test
	public void Benchmark_TicTacToe() {
		runTest("Benchmark_TicTacToe");
	}

	@Test
	public void Benchmark_Fibonacci() {
		runTest("Benchmark_Fibonacci");
	}

	@Test
	public void Benchmark_Queens() {
		runTest("Benchmark_Queens");
	}

	@Test
	public void Benchmark_MatrixMult() {
		runTest("Benchmark_MatrixMult");
	}

	@Test
	public void Benchmark_TrafficLights() {
		runTest("Benchmark_TrafficLights");
	}


}
