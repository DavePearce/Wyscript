package wyscript.testing;

import static org.junit.Assert.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import wyscript.io.Lexer;
import wyscript.io.Parser;
import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.ParFor;
import wyscript.lang.Type;
import wyscript.lang.WyscriptFile;
import wyscript.par.KernelWriter;
import wyscript.util.TypeChecker;

public class KernelWriterTests {

	@Test
	public void test() {
		Expr.Variable listVar = new Expr.Variable("a");
		Expr.Variable index = new Expr.Variable("i");
		Expr.IndexOf indexOf = new Expr.IndexOf(listVar, index);
		Stmt.Assign assign = new Stmt.Assign(indexOf, new Expr.Constant(1));
		List<Stmt> body = new ArrayList<Stmt>();
		//now configure it for the test
		Map<String,Type> environment = new HashMap<String,Type>();
		environment.put("a", new Type.List(new Type.Int()));
		body.add(assign);
		List<Expr> listContents = new ArrayList<Expr>();
		listContents.add(new Expr.Constant(1));
		//construct the for loop
		Stmt.ParFor forLoop = new Stmt.ParFor(index,
				new Expr.ListConstructor(listContents)
				, body);
		KernelWriter writer = new KernelWriter("test", environment, forLoop);
		System.out.println(writer.toString());
	}
	@Test
	public void test2() {
		String code = "void main():\n\t" +
				"int x = 0\n\t"
				+"[int] list = [1,2,3,4,5]\n\t"+
				"parFor i in 0..|list|:\n\t\t"+
				"x = list[i]\n";
		WyscriptFile tree = parseForTest(code,"testfile");
		WyscriptFile.FunDecl dec = tree.functions("main").get(0);
		Stmt.ParFor forloop = (Stmt.ParFor) dec.statements.get(2); //is a for loop
		//TypeChecker check = new TypeChecker();
		//check.check(tree);
		Map<String,Type> env = new HashMap<String,Type>();
		env.put("list", new Type.List(new Type.Int()));
		env.put("x", new Type.Int());
		env.put("i", new Type.Int());
		//check.check(forloop.getBody(),env);
		KernelWriter writer = new KernelWriter("test", env, forloop);
		System.out.println(writer);
	}
	@Test
	public void test3() {
		WyscriptFile tree = parseForTest("void main():\n"+
				"\t[int] a = [1,2,3,4,5]\n"+
				"\tparFor i in 0..|a|:\n"+
				"\t\ta[i] = a[i]+1","testfile"
				);
		WyscriptFile.FunDecl main = tree.functions("main").get(0);
		KernelWriter writer = new KernelWriter("test", new HashMap<String,Type>(),
				(Stmt.ParFor)main.statements.get(1));
		System.out.println(writer.toString());
		//TODO implement me, test this parallel for!
	}
	/**
	 * Quickly parse a string of WyScript
	 * @param content
	 * @return
	 */
	private WyscriptFile parseForTest(String content,String name) {
		BufferedReader reader = new BufferedReader(new StringReader(content));
		File srcFile = new File(name);
		Lexer lexer = null;
		try {
			lexer = new Lexer(reader);
		} catch (IOException e) {
			fail("Could not run test");
		}
		Parser parser = new Parser(name, lexer.scan());
		return parser.read();
	}
}
