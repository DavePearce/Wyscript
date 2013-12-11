package wyscript.par;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.sun.org.apache.xml.internal.utils.ObjectPool;

import wyscript.lang.Expr;
import wyscript.lang.Expr.*;
import wyscript.lang.Expr.BOp;
import wyscript.lang.Expr.IndexOf;
import wyscript.lang.Expr.LVal;
import wyscript.lang.Expr.Unary;
import wyscript.lang.Expr.Variable;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.ParFor;
import wyscript.lang.Type;
import wyscript.par.util.Argument;
import wyscript.par.util.LoopModule;
import wyscript.util.SyntacticElement;
import wyscript.util.SyntaxError.InternalFailure;


public class KernelWriter {
	private static final String NVCC_COMMAND = "/opt/cuda/bin/nvcc ";
	private ArrayList<Stmt> body;
	private Stmt.ParFor loop;

	private List<Argument> args;

	private String indexName1D = "i";
	private String indexName2D = "j";
	private String fileName;
	private String ptxFileName;
	private Map<String, Type> environment;
	private List<String> tokens;

	private boolean is2D;
	private LoopModule module;
	/**
	 * Initialise a KernelWriter which takes <i>name<i/> as its file name and uses
	 * the type mapping given in <i>environment</i> to generate the appropriate kernel
	 * for <i>loop</i>.
	 * @param filename
	 * @param environment
	 * @param loop
	 * @throws IOException
	 * @requires A correct mapping of the symbols used (when the parFor is executed) to their types
	 * @ensures All necessary parameters extracted and converted into a Cuda kernel, as well as stored within KernelWriter
	 */
	public KernelWriter(LoopModule module) {
		args = module.getArguments();
		is2D = module.is2D();
		this.module = module;
	}
	public void saveAndCompileKernel(String name , List<String> tokens) throws IOException {
		//save the token list to file
		File file = new File(name);
		FileWriter  writer = new FileWriter(file);
		for (String token : tokens) {
			if (token == null) {
				InternalFailure.internalFailure("Encountered null token", name, new Type.Void());
				writer.close();
				return;
			}
			writer.write(token);
			if (token.equals(";") || token.equals("{")) {
				writer.write("\n");
			}
			writer.write(" ");
		}
		writer.close();
		//now compile it into a ptx file
		preparePtxFile(name);
	}
	/**
	 *
	 * @param cuFileName
	 * @return
	 * @throws IOException
	 *
	 * @requires A well-formed Cuda file with a file extension.
	 * @ensures The Cuda file is compiled with the same name (without extension) as a .ptx file
	 */
    private String preparePtxFile(String cuFileName) throws IOException {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1) {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        this.ptxFileName = ptxFileName;

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            NVCC_COMMAND + " " + modelString + " -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;

        //System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        int exitValue = 0;
        try {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(
                "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0) {
        	///System.out.println(convertStreamToString(process.getErrorStream()));
        	InternalFailure.internalFailure("Failed to compile .ptx file for " +
        			"kernel. \nnvcc returned "+exitValue, cuFileName, loop);
        }

        //System.out.println("Finished creating PTX file");
        return ptxFileName;
    }
	public List<String> writeBody(ArrayList<Stmt> body , List<String> tokens ,
			Map<String,Type> environment) {
		this.environment = environment;
		writeThreadIndex(tokens);
		writeGuardBegin(tokens);
		for (Stmt statement : body) {
			write(statement,tokens);
		}
		tokens.add("}"); //add end of guard
		return tokens;
	}
	private void writeGuardBegin(List<String> tokens) {
		Expr src = loop.getSource();
		Expr.Binary srcBinary = (Expr.Binary) src;
		if (srcBinary.getOp().equals(Expr.BOp.RANGE)) {
			Expr low = srcBinary.getLhs();
			Expr high = srcBinary.getRhs();
			tokens.add("if");
			tokens.add("(");
			tokens.add(index1D());
			tokens.add("<");
			write(high,tokens);
			tokens.add("&&");
			tokens.add(index1D());
			tokens.add(">=");
			write(low,tokens);
			tokens.add(")");
			tokens.add("{");
		}else {
			InternalFailure.internalFailure("Expected loop source to be range " +
					"operator", fileName, src);
		}
	}
	private void writeThreadIndex(List<String> tokens) {
		//the 1D index
		if (!is2D) {
			tokens.add("int");
			tokens.add(index1D());
			tokens.add("=");
			tokens.add("blockIdx.x");
			tokens.add("*");
			tokens.add("blockDim.x");
			tokens.add("+");
			tokens.add("threadIdx.x");
			tokens.add(";");
		}else {
			//the 2D index
			tokens.add("int");
			tokens.add(index2D());
			tokens.add("=");
			tokens.add("blockIdx.x");
			tokens.add("*");
			tokens.add("blockDim.x");
			tokens.add("*");
			tokens.add("blockDim.y");
			tokens.add("+");
			tokens.add("threadIdx.y");
			tokens.add("*");
			tokens.add("blockDim.x");
			tokens.add("+");
			tokens.add("threadIdx.x");
			tokens.add(";");
		}
	}
	/**
	 * Convert a single statement to its appropriate kernel form. The statement must
	 * meet certain requirements of for conversion to Cuda code.
	 * @param statement
	 *
	 * @requires statement != null, statement is a legal and the parameters have been initialised
	 * @ensures A single statement of Cuda is written that correctly maps the Wyscript functionality
	 */
	private List<String> write(Stmt statement , List<String> tokens) {
		// what happens here?
		if (statement instanceof Stmt.IfElse) {
			write((Stmt.IfElse)statement,tokens);
		}else if (statement instanceof Stmt.VariableDeclaration) {
			write((Stmt.VariableDeclaration) statement,tokens);
		}else if (statement instanceof Stmt.Assign) {
			write((Stmt.Assign)statement,tokens);
		}else if (statement instanceof Stmt.ParFor) {
			write((Stmt.ParFor)statement,tokens);
		}
		else {
			InternalFailure.internalFailure("Encountered syntactic element not " +
					"supported in parFor loop", fileName, statement);
		}
		return tokens;
	}
	/**
	 * Writes an assignment statement to the kernel
	 * @param assign
	 *
	 * @requires
	 * @ensures
	 */
	private void write(Stmt.Assign assign, List<String> tokens) {
		Expr.LVal lhs = assign.getLhs();
		write(lhs,tokens);
		tokens.add("=");
		Expr rhs = assign.getRhs();
		write(rhs,tokens);
		tokens.add(";");
	}
	/**
	 * Writes a single expression to the kernel
	 * @param expression
	 *
	 * @requires expression is of an acceptable type and has appropriate parameters
	 * @ensures the Cuda form of the expression is written to the token list
	 */
	private void write(Expr expression,List<String> tokens) {
		if (expression instanceof Expr.ListConstructor) {
			write((Expr.ListConstructor) expression,tokens);
		}
		else if (expression instanceof Expr.LVal) {
			write((Expr.LVal)expression,tokens);
		}else if (expression instanceof Expr.Variable) {
			write((Expr.Variable) expression,tokens);
		}else if (expression instanceof Expr.Constant) {
			write((Expr.Constant) expression,tokens);
		}else if (expression instanceof Expr.IndexOf) {
			write((Expr.IndexOf)expression,tokens);
		}else if (expression instanceof Expr.Binary) {
			write((Expr.Binary) expression,tokens);
		}else if (expression instanceof Expr.Unary) {
			write((Expr.Unary)expression,tokens);
		}else if ((expression instanceof Expr.ListConstructor)) {
			write((Expr.ListConstructor)expression,tokens);
		}
		else{
			InternalFailure.internalFailure("Could not write expression to kernel. Unknown expresion type", fileName, expression);
		}
	}
	/**
	 * Writes a single Expr.LVal to the kernel.
	 * @param val
	 *
	 * @ensures An expression is written with the correct referencing
	 * and referencing of pointers.
	 */
	private void write(Expr.LVal val,List<String> tokens) {
		if (val instanceof Expr.Variable) {
			//if this is a parameter, have to dereference the pointer
			if (module.isParameter(((Expr.Variable)val).getName())) {
					tokens.add("*");
			}
			//write a
			Expr.Variable variable = (Expr.Variable) val;
			//simply add the variable name
			String name = variable.getName();
			if (name.equals(loop.getIndex().getName())) {
				tokens.add(index1D());
			}else {
				tokens.add(name);
			}
		}else if (val instanceof Expr.IndexOf) {
			write((Expr.IndexOf)val,tokens);
		}

	}
	/**
	 * Here a nested loop is being written
	 * @param statement
	 * @param tokens
	 */
	private void write(Stmt.ParFor statement, List<String> tokens) {

	}
	private String index1D() {
		return indexName1D;
	}
	private void write(Expr.Binary binary,List<String> tokens) {
		tokens.add("(");
		write(binary.getLhs(),tokens);
		tokens.add(")");
		writeOp(binary.getOp(),tokens);
		tokens.add("(");
		write(binary.getRhs(),tokens);
		tokens.add(")");
	}
	private void write(Expr.Unary unary,List<String> tokens) {
		switch (unary.getOp()) {
		case LENGTHOF:
			writeLengthOf(unary.getExpr(),tokens);
			break;
		case NEG:
			tokens.add("-");
			tokens.add("(");
			write(unary.getExpr(),tokens);
			tokens.add(")");
			break;
		case NOT:
			tokens.add("!");
			tokens.add("(");
			write(unary.getExpr(),tokens);
			tokens.add(")");
			break;
		default:
			InternalFailure.internalFailure("Unknown unary expression encountered"
					, fileName, unary);

		}
	}
	private void writeLengthOf(Expr expr,List<String> tokens) {
		if (expr instanceof Expr.Variable) {
			String name = ((Expr.Variable)expr).getName();
			tokens.add("(");
			tokens.add("*");
			String lengthName = lengthMap.get(name);
			tokens.add(lengthName);
			tokens.add(")");
		}
		else if (expr instanceof Expr.ListConstructor) {
			int size = ((Expr.ListConstructor) expr).getArguments().size();
			tokens.add(Integer.toString(size));
		}
		else {
			//TODO Implement me
			InternalFailure.internalFailure("Writing length of this expression not implemented", fileName, expr);
		}
	}
	private void writeOp(BOp op,List<String> tokens) {
		tokens.add(op.toString());
	}
	/**
	 * Writes a single constant to token list. This constant may be an int only
	 * @param constant
	 *
	 * @requires the constant is an int
	 */
	private void write(Expr.Constant constant,List<String> tokens) {
		Object val = constant.getValue();
		if (val instanceof Integer) {
			tokens.add(Integer.toString((Integer)val));
		}else {
			InternalFailure.internalFailure("Cannot write this constant: "+val,
					fileName, constant);
		}
	}
	/**
	 * This method is not implemented yet.
	 *
	 * Writes a list constructor to kernel code.
	 * @param list
	 */
	private void write(Expr.ListConstructor list,List<String> tokens) {
		tokens.add("{");
		boolean needComma = false;
		for (Expr expr : list.getArguments()) {
			if (needComma) tokens.add(",");
			write(expr,tokens);
			needComma = true;
		}
		tokens.add("}");
/*		InternalFailure.internalFailure("Writing list constructors not implemented"
				, fileName, list);*/
	}
	/**
	 * Write a single indexOf operation to token list
	 * @param expr
	 */
	private void write(Expr.IndexOf indexOf, List<String> tokens) {
		if (indexOf.getSource() instanceof Expr.Variable) {
			Expr src = indexOf.getSource();
			Expr indexVar = indexOf.getIndex();
			//indexVar is an instance of [int]
			//source expression must be of type...
			if (src instanceof Expr.Variable) {
				write1DIndexOf(tokens, (Expr.Variable) src, indexVar);
			}else if (src instanceof Expr.IndexOf) {
				write2DIndexOf(tokens, (IndexOf) src, indexVar);
			}
			else {
				InternalFailure.internalFailure("Can only perform indexof on list", fileName, indexOf);
			}
		}else {
			InternalFailure.internalFailure("Expected source type to be of type list", fileName, indexOf.getSource());
		}
	}
	/**
	 *
	 * @param tokens
	 * @param src
	 * @param indexVar
	 */
	private void write2DIndexOf(List<String> tokens, Expr.IndexOf src, Expr indexVar) {
		Expr indexSrc = src.getSource();
		if (indexSrc instanceof Expr.Variable) {

		}else {
			//there is an issue if this happens
		}
	}
	private String index2D() {
		return indexName2D;
	}
	/**
	 *
	 * @param tokens
	 * @param src
	 * @param indexVar
	 */
	private void write1DIndexOf(List<String> tokens, Expr.Variable src, Expr indexVar) {
		Type typeOfVar = environment.get(src.getName());
		Type listType = ((Type.List)typeOfVar).getElement();
		if (listType instanceof Type.Int) {
			//the type is correct for a kernel, write it here
			tokens.add(src.getName());
			if (indexVar instanceof Expr.Variable) {
				if (((Expr.Variable) indexVar).getName().equals(loop.getIndex().getName())) {
					if (!is1D) InternalFailure.internalFailure("Expected to index 1D list", fileName, indexVar);
					tokens.add("["+index1D()+"]");
				}
			}else if (indexVar instanceof Expr.Constant) {
				tokens.add("[");
				write((Expr.Constant)indexVar,tokens);
				tokens.add("]");
			}else {
				InternalFailure.internalFailure("Index should be parFor loop index or constant", fileName, src);
			}
		}
		else{
			InternalFailure.internalFailure("List type should be int for kernel conversion", fileName, src);
		}
	}
	/**
	 * Writes a classical conditional statement to the kernel
	 * @param statement
	 */
	private void write(Stmt.IfElse statement,List<String> tokens) {
		tokens.add("if");
		tokens.add("(");
		//the condition can only be simple equality, or a statement in boolean
		//logic
		Expr expression = statement.getCondition();
		write(expression,tokens);
		tokens.add(")");
		tokens.add("{");
		//branches may be empty
		for (Stmt s : statement.getTrueBranch()) {
			write(s,tokens); //write the single statement
		}
		tokens.add("}");
		tokens.add("else");
		tokens.add("{");
		for (Stmt s : statement.getFalseBranch()) {
			write(s,tokens); //write the single statement
		}
		tokens.add("}");
	}
	/**
	 * Writes a single variable declaration to the kernel.
	 * @param decl
	 */
	private void write(Stmt.VariableDeclaration decl,List<String> tokens) {
		Type type = decl.getType();
		if (type instanceof Type.Int) {
			tokens.add("int");
			tokens.add(decl.getName());
			tokens.add("=");
			//now write the expression
			write(decl.getExpr(),tokens);
			tokens.add(";");
		}else if (type instanceof Type.List) {
			Type element = ((Type.List) type).getElement();
			if (element instanceof Type.Int && decl.getExpr() instanceof Expr.ListConstructor) {
				List<Expr> elements = ((Expr.ListConstructor)decl.getExpr()).getArguments();
				tokens.add("int");
				tokens.add(decl.getName());
				tokens.add("["+elements.size()+"]");
				tokens.add("=");
				write((Expr.ListConstructor)decl.getExpr(),tokens);
				tokens.add(";");
			}else {
				InternalFailure.internalFailure("Can only write explicit list of integers",fileName,decl);
			}
		}
		else {
			InternalFailure.internalFailure("Cannot write variable declaration for the given type",fileName,decl);

		}
		environment.put(decl.getName(), type);
	}
	/**
	 * Return the File object associated with this kernel
	 * @return
	 */
	public File getPtxFile() {
		return new File(ptxFileName);
	}
	/**
	 * Returns a List of the string representation of the kernel writer's tokens
	 * @return
	 */
	public List<String> getTokenList() {
		List<String> output = new ArrayList<String>(tokens);
		return output;
	}
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		for (String t : tokens) {
			builder.append(t);
			builder.append(" ");
			if (t.equals(";")) {
				builder.append("\n");
			}
		}
		return builder.toString();
	}
	public SyntacticElement getLoop() {
		return loop;
	}
	public Map<String, Type> getEnvironment() {
		return environment;
	}
}