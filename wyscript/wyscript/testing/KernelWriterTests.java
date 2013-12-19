package wyscript.testing;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.par.KernelWriter;

public class KernelWriterTests {

	@Test
	public void test() {
		fail("Not yet implemented");
	}
	@Test
	public void test_intDec() {
		Expr.Constant constant = new Expr.Constant(1);
		List<Stmt> bod = new ArrayList<Stmt>();
		Stmt.VariableDeclaration declaration = new Stmt.VariableDeclaration(new Type.Int(), "x", constant);
		bod.add(declaration);
		Stmt.For loop = new Stmt.For(null, null, null, bod);
		KernelWriter writer = new KernelWriter(loop);
		List<String> testTokens = writer.getTokenList();
		checkElementsEqual(new String[]{"int","x","=","1"},testTokens.toArray());
	}
	private void checkElementsEqual(Object[] array1, Object[] array2) {
		if (array2.length == array1.length) {
			for (int i = 0; i < array1.length ; i++) {
				if (!array1[i].equals(array2[i])) {
					System.out.println("Arrays not equal");
					System.out.println("array1 "+array1.toString());
					System.out.println("array2 "+array2.toString());
					fail("array1 has different element to array2 at index "+i);
				}
			}
		}else{
			fail("array 1 had length "+array1.length +" whilst array 2 had length "
		+array2.length);
		}
	}
}
