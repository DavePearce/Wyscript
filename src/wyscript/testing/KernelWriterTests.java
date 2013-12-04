package wyscript.testing;

import static org.junit.Assert.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.junit.Test;

import wyscript.io.Lexer;
import wyscript.io.Parser;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.ParFor;
import wyscript.lang.Type;
import wyscript.lang.WyscriptFile;
import wyscript.par.KernelWriter;

public class KernelWriterTests {
	private final String testDir = "/am/state-opera/home1/antunomate/summer_research/wy_material/WyScript_fork/Wyscript/partests/";

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
	/**
	 * Quickly parse a wyscript file
	 * @param content
	 * @return
	 */
	private WyscriptFile parseForFile(String filename) {
		Lexer lexer = null ;
		try {
			lexer = new Lexer(filename);
		} catch (IOException e) {
			fail("Could not lex file.");
		}
		Parser parser = new Parser(filename, lexer.scan());
		return parser.read();
	}
	private void performTest(String testName , Map<String,Type> environment) {
		//sort out the files
		String wyPath = testDir+testName+".wys";
		String kernelPath = testDir+testName+".cu";
		File kernelFile = new File(kernelPath);
		Scanner cuScan = null;
		WyscriptFile wyFile = parseForFile(wyPath);
		ParFor loop = getFirstLoop(wyFile.functions("main").get(0));

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
				System.out.println("Unexpectedly reached end of token list");
				System.out.println("Kernel writer gave this:");
				printProg(tokens);
				fail("Reached end of kernel writer output before scan of file complete.");
			}
			String token = cuScan.next();
			if (tokens.get(i).equals(token)) {
				cuSaved.add(token);
				writerSaved.add(token);
				writerSaved.add(tokens.get(i));
			}
			i++;
		}
		if (i<tokens.size()) {
			System.out.println("Reached end of .cu output before writer output");
			System.out.println("Kernel writer gave this:");
			printProg(tokens);
			System.out.println("Problematic writer token is '"+tokens.get(i)+"' at index "+i);
//			System.out.println("Got to this in writer output before failure:");
//			printProg(writerSaved);
			fail("Reached end of .cu file before scan of kernel writer output complete.");
		}
	}
	private void printProg(List<String> tokens) {
		StringBuilder builder = new StringBuilder();
		for (String str : tokens) {
			builder.append(str);
			builder.append(" ");
		}
		System.out.println(builder.toString());
	}
	private ParFor getFirstLoop(WyscriptFile.FunDecl function) {
		for (Stmt stmt : function.statements) {
			if (stmt instanceof Stmt.ParFor) return (ParFor) stmt;
		}
		fail("Could not find parFor loop in function");
		return null; //unreachable
	}
}
