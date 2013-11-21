package wyscript.testing;

import java.io.File;

import org.junit.*;


public class ClassFileValidTests {
	/**
	 * Path to test directory.
	 */
	private String testdir = "tests/valid/";

	// ======================================================================
	// Test Harness
	// ======================================================================

	private void runTest(String name) {		
		// The name of the file which contains the output for this test
		String sampleOutputFile = testdir + File.separatorChar + name
				+ ".sysout";

		// Classpath to project root
		String classPath = "../../src";
		
		// First, we need to compile the given test into javascript
		TestUtils.exec(classPath, testdir, "whilelang.Main", "-jvm", name + ".while");
		
		// Second, execute the generated JavaScript Program. 
		String output = TestUtils.exec(testdir,testdir,name);

		// Third, compare the output!
		TestUtils.compare(output,sampleOutputFile);
	}

	// ======================================================================
	// Tests
	// ======================================================================

	@Test
	public void BoolAssign_Valid_1() {
		runTest("BoolAssign_Valid_1");
	}

	@Test
	public void BoolAssign_Valid_2() {
		runTest("BoolAssign_Valid_2");
	}

	@Test
	public void BoolAssign_Valid_3() {
		runTest("BoolAssign_Valid_3");
	}

	@Test
	public void BoolAssign_Valid_4() {
		runTest("BoolAssign_Valid_4");
	}

	@Test
	public void BoolIfElse_Valid_1() {
		runTest("BoolIfElse_Valid_1");
	}

	@Test
	public void BoolIfElse_Valid_2() {
		runTest("BoolIfElse_Valid_2");
	}

	@Test
	public void BoolList_Valid_1() {
		runTest("BoolList_Valid_1");
	}

	@Test
	public void BoolList_Valid_2() {
		runTest("BoolList_Valid_2");
	}

	@Test
	public void BoolRecord_Valid_1() {
		runTest("BoolRecord_Valid_1");
	}

	@Test
	public void BoolRecord_Valid_2() {
		runTest("BoolRecord_Valid_2");
	}

	@Test
	public void BoolReturn_Valid_1() {
		runTest("BoolReturn_Valid_1");
	}

	@Test
	public void Cast_Valid_1() {
		runTest("Cast_Valid_1");
	}

	@Test
	public void Cast_Valid_2() {
		runTest("Cast_Valid_2");
	}

	@Test
	public void Cast_Valid_3() {
		runTest("Cast_Valid_3");
	}

	@Test
	public void Cast_Valid_4() {
		runTest("Cast_Valid_4");
	}

	@Test
	public void Char_Valid_1() {
		runTest("Char_Valid_1");
	}

	@Test
	public void Char_Valid_2() {
		runTest("Char_Valid_2");
	}

	@Test
	public void Char_Valid_3() {
		runTest("Char_Valid_3");
	}

	@Test
	public void Const_Valid_1() {
		runTest("Const_Valid_1");
	}

	@Test
	public void Const_Valid_2() {
		runTest("Const_Valid_2");
	}

	@Test
	public void Const_Valid_3() {
		runTest("Const_Valid_3");
	}

	@Test
	public void Const_Valid_4() {
		runTest("Const_Valid_4");
	}

	@Test
	public void Define_Valid_1() {
		runTest("Define_Valid_1");
	}

	@Test
	public void Define_Valid_2() {
		runTest("Define_Valid_2");
	}

	@Test
	public void Function_Valid_1() {
		runTest("Function_Valid_1");
	}

	@Test
	public void Function_Valid_2() {
		runTest("Function_Valid_2");
	}

	@Test
	public void Function_Valid_4() {
		runTest("Function_Valid_4");
	}

	@Test
	public void IfElse_Valid_1() {
		runTest("IfElse_Valid_1");
	}

	@Test
	public void IfElse_Valid_2() {
		runTest("IfElse_Valid_2");
	}

	@Test
	public void IfElse_Valid_3() {
		runTest("IfElse_Valid_3");
	}

	@Test
	public void IntDefine_Valid_1() {
		runTest("IntDefine_Valid_1");
	}

	@Test
	public void IntDiv_Valid_1() {
		runTest("IntDiv_Valid_1");
	}

	@Test
	public void IntDiv_Valid_2() {
		runTest("IntDiv_Valid_2");
	}

	@Test
	public void IntEquals_Valid_1() {
		runTest("IntEquals_Valid_1");
	}

	@Test
	public void IntMul_Valid_1() {
		runTest("IntMul_Valid_1");
	}

	@Test
	public void LengthOf_Valid_1() {
		runTest("LengthOf_Valid_1");
	}

	@Test
	public void LengthOf_Valid_5() {
		runTest("LengthOf_Valid_5");
	}

	@Test
	public void ListAccess_Valid_1() {
		runTest("ListAccess_Valid_1");
	}

	@Test
	public void ListAccess_Valid_3() {
		runTest("ListAccess_Valid_3");
	}

	@Test
	public void ListAccess_Valid_4() {
		runTest("ListAccess_Valid_4");
	}

	@Test
	public void ListAppend_Valid_1() {
		runTest("ListAppend_Valid_1");
	}

	@Test
	public void ListAppend_Valid_2() {
		runTest("ListAppend_Valid_2");
	}

	@Test
	public void ListAppend_Valid_3() {
		runTest("ListAppend_Valid_3");
	}

	@Test
	public void ListAppend_Valid_4() {
		runTest("ListAppend_Valid_4");
	}

	@Test
	public void ListAppend_Valid_5() {
		runTest("ListAppend_Valid_5");
	}

	@Test
	public void ListAppend_Valid_6() {
		runTest("ListAppend_Valid_6");
	}

	@Test
	public void ListAppend_Valid_7() {
		runTest("ListAppend_Valid_7");
	}

	@Test
	public void ListAssign_Valid_1() {
		runTest("ListAssign_Valid_1");
	}

	@Test
	public void ListAssign_Valid_2() {
		runTest("ListAssign_Valid_2");
	}

	@Test
	public void ListAssign_Valid_3() {
		runTest("ListAssign_Valid_3");
	}

	@Test
	public void ListAssign_Valid_4() {
		runTest("ListAssign_Valid_4");
	}

	@Test
	public void ListAssign_Valid_5() {
		runTest("ListAssign_Valid_5");
	}

	@Test
	public void ListAssign_Valid_6() {
		runTest("ListAssign_Valid_6");
	}

	@Test
	public void ListAssign_Valid_10() {
		runTest("ListAssign_Valid_10");
	}

	@Test
	public void ListConversion_Valid_1() {
		runTest("ListConversion_Valid_1");
	}

	@Test
	public void ListEmpty_Valid_1() {
		runTest("ListEmpty_Valid_1");
	}

	@Test
	public void ListEquals_Valid_1() {
		runTest("ListEquals_Valid_1");
	}

	@Test
	public void ListGenerator_Valid_1() {
		runTest("ListGenerator_Valid_1");
	}

	@Test
	public void ListGenerator_Valid_2() {
		runTest("ListGenerator_Valid_2");
	}

	@Test
	public void ListGenerator_Valid_3() {
		runTest("ListGenerator_Valid_3");
	}

	@Test
	public void ListLength_Valid_1() {
		runTest("ListLength_Valid_1");
	}

	@Test
	public void ListLength_Valid_2() {
		runTest("ListLength_Valid_2");
	}

	@Test
	public void MultiLineComment_Valid_1() {
		runTest("MultiLineComment_Valid_1");
	}

	@Test
	public void MultiLineComment_Valid_2() {
		runTest("MultiLineComment_Valid_2");
	}

	@Test
	public void RealDiv_Valid_1() {
		runTest("RealDiv_Valid_1");
	}

	@Test
	public void RealDiv_Valid_3() {
		runTest("RealDiv_Valid_3");
	}

	@Test
	public void RealDiv_Valid_4() {
		runTest("RealDiv_Valid_4");
	}

	@Test
	public void RealNeg_Valid_1() {
		runTest("RealNeg_Valid_1");
	}

	@Test
	public void RealSub_Valid_1() {
		runTest("RealSub_Valid_1");
	}

	@Test
	public void RealSub_Valid_2() {
		runTest("RealSub_Valid_2");
	}

	@Test
	public void Real_Valid_1() {
		runTest("Real_Valid_1");
	}

	@Test
	public void RecordAccess_Valid_2() {
		runTest("RecordAccess_Valid_2");
	}

	@Test
	public void RecordAssign_Valid_1() {
		runTest("RecordAssign_Valid_1");
	}

	@Test
	public void RecordAssign_Valid_2() {
		runTest("RecordAssign_Valid_2");
	}

	@Test
	public void RecordAssign_Valid_3() {
		runTest("RecordAssign_Valid_3");
	}

	@Test
	public void RecordAssign_Valid_4() {
		runTest("RecordAssign_Valid_4");
	}

	@Test
	public void RecordAssign_Valid_5() {
		runTest("RecordAssign_Valid_5");
	}

	@Test
	public void RecordAssign_Valid_6() {
		runTest("RecordAssign_Valid_6");
	}

	@Test
	public void RecordDefine_Valid_1() {
		runTest("RecordDefine_Valid_1");
	}

	@Test
	public void Remainder_Valid_1() {
		runTest("Remainder_Valid_1");
	}

	@Test
	public void SingleLineComment_Valid_1() {
		runTest("SingleLineComment_Valid_1");
	}

	@Test
	public void String_Valid_1() {
		runTest("String_Valid_1");
	}

	@Test
	public void String_Valid_2() {
		runTest("String_Valid_2");
	}

	@Test
	public void String_Valid_3() {
		runTest("String_Valid_3");
	}

	@Test
	public void String_Valid_4() {
		runTest("String_Valid_4");
	}

	@Test
	public void Switch_Valid_1() {
		runTest("Switch_Valid_1");
	}

	@Test
	public void Switch_Valid_2() {
		runTest("Switch_Valid_2");
	}

	@Test
	public void Switch_Valid_3() {
		runTest("Switch_Valid_3");
	}

	@Test
	public void Switch_Valid_4() {
		runTest("Switch_Valid_4");
	}

	@Test
	public void Switch_Valid_6() {
		runTest("Switch_Valid_6");
	}

	@Test
	public void Switch_Valid_7() {
		runTest("Switch_Valid_7");
	}

	@Test
	public void Switch_Valid_8() {
		runTest("Switch_Valid_8");
	}

	@Test
	public void TypeEquals_Valid_1_RuntimeTest() {
		runTest("TypeEquals_Valid_1");
	}

	@Test
	public void TypeEquals_Valid_2_RuntimeTest() {
		runTest("TypeEquals_Valid_2");
	}

	@Test
	public void TypeEquals_Valid_5_RuntimeTest() {
		runTest("TypeEquals_Valid_5");
	}

	@Test
	public void TypeEquals_Valid_8_RuntimeTest() {
		runTest("TypeEquals_Valid_8");
	}

	@Test
	public void TypeEquals_Valid_9_RuntimeTest() {
		runTest("TypeEquals_Valid_9");
	}

	@Test
	public void TypeEquals_Valid_11_RuntimeTest() {
		runTest("TypeEquals_Valid_11");
	}

	@Test
	public void TypeEquals_Valid_14_RuntimeTest() {
		runTest("TypeEquals_Valid_14");
	}

	@Test
	public void TypeEquals_Valid_16_RuntimeTest() {
		runTest("TypeEquals_Valid_16");
	}

	@Test
	public void TypeEquals_Valid_20_RuntimeTest() {
		runTest("TypeEquals_Valid_20");
	}

	@Test
	public void UnionType_Valid_1() {
		runTest("UnionType_Valid_1");
	}

	@Test
	public void UnionType_Valid_2() {
		runTest("UnionType_Valid_2");
	}

	@Test
	public void UnionType_Valid_4() {
		runTest("UnionType_Valid_4");
	}

	@Test
	public void UnionType_Valid_5() {
		runTest("UnionType_Valid_5");
	}

	@Test
	public void UnionType_Valid_6() {
		runTest("UnionType_Valid_6");
	}

	@Test
	public void UnionType_Valid_7() {
		runTest("UnionType_Valid_7");
	}

	@Test
	public void UnionType_Valid_8() {
		runTest("UnionType_Valid_8");
	}

	@Test
	public void UnionType_Valid_9() {
		runTest("UnionType_Valid_9");
	}

	@Test
	public void While_Valid_1() {
		runTest("While_Valid_1");
	}

	@Test
	public void While_Valid_2() {
		runTest("While_Valid_2");
	}

	@Test
	public void While_Valid_4() {
		runTest("While_Valid_4");
	}

	@Test
	public void While_Valid_6() {
		runTest("While_Valid_6");
	}
}
