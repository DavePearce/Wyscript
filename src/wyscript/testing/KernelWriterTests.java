package wyscript.testing;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.par.KernelWriter;

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

}
