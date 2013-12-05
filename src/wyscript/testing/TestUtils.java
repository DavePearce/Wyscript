package wyscript.testing;

import static org.junit.Assert.fail;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import wyscript.io.Lexer;
import wyscript.io.Parser;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.lang.WyscriptFile;
import wyscript.lang.Stmt.ParFor;
import wyscript.par.KernelWriter;
import wyscript.util.TypeChecker;

/**
 * Provides some simple helper functions used by all test harnesses.
 *
 * @author David J. Pearce
 *
 */
public class TestUtils {
	private static final String kernelWriteTestDir = "/am/state-opera/home1/antunomate/summer_research/wy_material/WyScript_fork/Wyscript/partests/";
	private static final String JCUDA_HOME = "/home/state-opera1/antunomate/summer_research/jcuda_bin/JCuda-All-0.5.0b-bin-linux-x86_64";

	/**
	 * Execute a given class file using the "java" command, and return all
	 * output written to stdout. In the case of some kind of failure, write the
	 * generated stderr stream to this processes stdout.
	 *
	 * @param classPath
	 *            Class path to use when executing Java code. Note, directories
	 *            can always be safely separated with '/', and path separated
	 *            with ':'.
	 * @param srcDir
	 *            Path to root of package containing class. Note, directories
	 *            can always be safely separated with '/'.
	 * @param className
	 *            Name of class to execute
	 * @param args
	 *            Arguments to supply on the command-line.
	 * @return All output generated from the class that was written to stdout.
	 */
	public static String exec(String classPath, String srcDir, String className, String... args) {
		try {
			classPath = classPath.replace('/', File.separatorChar);
			classPath = classPath.replace(':', File.pathSeparatorChar);
			srcDir = srcDir.replace('/', File.separatorChar);
			String tmp = "java -Djava.library.path="+JCUDA_HOME+" -cp " + classPath + " " + className;
			for (String arg : args) {
				tmp += " " + arg;
			}
			Process p = Runtime.getRuntime().exec(tmp, null, new File(srcDir));

			StringBuffer syserr = new StringBuffer();
			StringBuffer sysout = new StringBuffer();
			new StreamGrabber(p.getErrorStream(), syserr);
			new StreamGrabber(p.getInputStream(), sysout);
			int exitCode = p.waitFor();
			if (exitCode != 0) {
				System.err
						.println("============================================================");
				System.err.println(className);
				System.err
						.println("============================================================");
				System.err.println(syserr);
				return null;
			} else {
				return sysout.toString();
			}
		} catch (Exception ex) {
			ex.printStackTrace();
			fail("Problem running compiled test");
		}

		return null;
	}

	/**
	 * Compare the output of executing java on the test case with a reference
	 * file. If the output differs from the reference output, then the offending
	 * line is written to the stdout and an exception is thrown.
	 *
	 * @param output
	 *            This provides the output from executing java on the test case.
	 * @param referenceFile
	 *            The full path to the reference file. This should use the
	 *            appropriate separator char for the host operating system.
	 */
	public static void compare(String output, String referenceFile) {
		try {
			BufferedReader outReader = new BufferedReader(new StringReader(
					output));
			BufferedReader refReader = new BufferedReader(new FileReader(
					new File(referenceFile)));

			while (refReader.ready() && outReader.ready()) {
				String a = refReader.readLine();
				String b = outReader.readLine();

				if (a.equals(b)) {
					continue;
				} else {
					System.err.println(" > " + a);
					System.err.println(" < " + b);
					throw new Error("Output doesn't match reference");
				}
			}

			String l1 = outReader.readLine();
			String l2 = refReader.readLine();
			if (l1 == null && l2 == null)
				return;
			do {
				l1 = outReader.readLine();
				l2 = refReader.readLine();
				if (l1 != null) {
					System.err.println(" < " + l1);
				} else if (l2 != null) {
					System.err.println(" > " + l2);
				}
			} while (l1 != null && l2 != null);

			fail("Files do not match");
		} catch (Exception ex) {
			ex.printStackTrace();
			fail();
		}
	}

	/**
	 * Grab everything produced by a given input stream until the End-Of-File
	 * (EOF) is reached. This is implemented as a separate thread to ensure that
	 * reading from other streams can happen concurrently. For example, we can
	 * read concurrently from <code>stdin</code> and <code>stderr</code> for
	 * some process without blocking that process.
	 *
	 * @author David J. Pearce
	 *
	 */
	static public class StreamGrabber extends Thread {
		private InputStream input;
		private StringBuffer buffer;

		StreamGrabber(InputStream input, StringBuffer buffer) {
			this.input = input;
			this.buffer = buffer;
			start();
		}

		public void run() {
			try {
				int nextChar;
				// keep reading!!
				while ((nextChar = input.read()) != -1) {
					buffer.append((char) nextChar);
				}
			} catch (IOException ioe) {
			}
		}
	}
	///////// MATE'S UTILS BEGIN HERE
	/**
	 * Quickly parse a wyscript file
	 * @param content
	 * @return
	 */
	public static WyscriptFile parseForFile(String filename) {
		Lexer lexer = null ;
		try {
			lexer = new Lexer(filename);
		} catch (IOException e) {
			fail("Could not lex file. "+e.getMessage());
		}
		Parser parser = new Parser(filename, lexer.scan());
		return parser.read();
	}
	public static void writerTest(String testName , Map<String,Type> environment) {
		//sort out the files
		String wyPath = kernelWriteTestDir+testName+".wys";
		String kernelPath = kernelWriteTestDir+testName+".cu";
		File kernelFile = new File(kernelPath);
		Scanner cuScan = null;
		WyscriptFile wyFile = parseForFile(wyPath);
		ParFor loop = getFirstLoop(wyFile.functions("main").get(0));

		doKernelTest(testName, environment, kernelFile, cuScan, loop);
	}

	private static void doKernelTest(String testName, Map<String, Type> environment,
			File kernelFile, Scanner cuScan, ParFor loop) {
		try {
			cuScan = new Scanner(kernelFile);
		} catch (FileNotFoundException e) {
			fail("Test could not be performed");
		}
		KernelWriter writer = new KernelWriter(testName,environment,loop);
		List<String> tokens = writer.getTokenList();
		List<String> cuSaved = new ArrayList<String>();
		List<String> writerSaved = new ArrayList<String>();
		int i = 0;
		while (cuScan.hasNext()) {
			if (i>=tokens.size())  {
				System.out.println("Unexpectedly reached end of kernel writer token list");
				System.out.println("Kernel writer gave this:");
				printProg(tokens);
				fail("Reached end of kernel writer output before scan of file complete.");
			}
			String token = cuScan.next();
			if (tokens.get(i).equals(token)) {
				cuSaved.add(tokens.get(i));
				writerSaved.add(tokens.get(i));
				//writerSaved.add(tokens.get(i));
			}else {
				System.out.println("Kernel writer gave this:");
				printProg(tokens);
				System.out.println("Got up to here: ");
				printProg(cuSaved);
				fail("Tokens do not match: expected '"+token+"' got '"+tokens.get(i)+"'");
			}
			i++;
		}
		if (i<tokens.size()) {
			System.out.println("Reached end of .cu output before writer output");
			System.out.println("Kernel writer gave this:");
			printProg(tokens);
			System.out.println("Problematic writer token is '"+tokens.get(i)+"' at index "+i);
			fail("Reached end of .cu file before scan of kernel writer output complete.");
		}
	}
	public static void writerTest(String testName) {
		TypeChecker checker = new TypeChecker();
		String wyPath = kernelWriteTestDir+testName+".wys";
		String kernelPath = kernelWriteTestDir+testName+".cu";
		File kernelFile = new File(kernelPath);
		Scanner cuScan = null;
		WyscriptFile wyFile = parseForFile(wyPath);
		ParFor loop = getFirstLoop(wyFile.functions("main").get(0));
		checker.check(wyFile);
		Map<String,Type> environment = checker.check(wyFile.functions("main").get(0));

		doKernelTest(testName, environment, kernelFile, cuScan, loop);
	}
	public static void printProg(List<String> tokens) {
		StringBuilder builder = new StringBuilder();
		for (String str : tokens) {
			builder.append(str);
			builder.append(" ");
		}
		System.out.println(builder.toString());
	}
	public static ParFor getFirstLoop(WyscriptFile.FunDecl function) {
		for (Stmt stmt : function.statements) {
			if (stmt instanceof Stmt.ParFor) return (ParFor) stmt;
		}
		fail("Could not find parFor loop in function");
		return null; //unreachable
	}
}
