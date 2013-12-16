package wyscript.par;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import wyscript.lang.Expr;
import wyscript.lang.Expr.BOp;
import wyscript.lang.Expr.IndexOf;
import wyscript.lang.Expr.Variable;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.par.util.Argument;
import wyscript.par.util.Category;
import wyscript.par.util.LoopModule;
import wyscript.util.SyntacticElement;
import wyscript.util.SyntaxError.InternalFailure;


public class OldKernelWriter {
	private static final String NVCC_COMMAND = "/opt/cuda/bin/nvcc ";
	private Stmt.ParFor loop;

	private String indexName1D = "i";
	private String indexName2D = "j";
	private String fileName;
	private String ptxFileName;
	private Map<String, Type> environment;
	private List<String> tokens = new ArrayList<String>();

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
	public OldKernelWriter(LoopModule module) {
		module.getArguments();
		this.module = module;
		module.getOuterLoop().getIndex();
		this.environment = module.getEnvironment();
		this.loop = module.getOuterLoop();
		this.fileName = module.getName();
		writeAll();
		try {
			saveAndCompileKernel(fileName, tokens);
		} catch (IOException e) {
			InternalFailure.internalFailure("Could not write kernel. Got error: "+e.getMessage()
					, fileName, module.getOuterLoop());
		}
	}
	private void writeAll() {
		tokens.add("extern");
		tokens.add("\"C\"");
		writeFunctionDeclaration(tokens, module.getArguments());
		tokens.add("{");
		ArrayList<Stmt> body = module.getOuterLoop().getBody();
		writeBody(body, tokens, environment);
		tokens.add("}");
	}
	public void saveAndCompileKernel(String name , List<String> tokens) throws IOException {
		//save the token list to file
		String cuName = name+".cu";
		File file = new File(cuName);
		FileWriter  writer = new FileWriter(file);
		for (String token : tokens) {
			if (token == null) {
				InternalFailure.internalFailure("Encountered null token", cuName, new Type.Void());
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
		preparePtxFile(cuName);
	}
	public List<String> writeFunctionDeclaration(List<String> tokens , List<Argument> arguments) {
    	tokens.add("__global__");
    	tokens.add("void");
    	tokens.add(getFunctionName());
    	tokens.add("(");
    	for (int i = 0 ; i < arguments.size() ; i++) {
    		Argument arg = arguments.get(i);
            if (i>=1 && i < arguments.size()) {
                tokens.add(",");
            }
            tokens.add(arg.getCType());
            tokens.add(convertName(arg));
    	}
    	tokens.add(")");
		return tokens;
    }
	private String convertName(Argument arg) {
		if (arg instanceof Argument.Length1D) {
			return arg.name + "_length";
		}else if(arg instanceof Argument.Length2D){
			if (((Argument.Length2D) arg).isHeight){
				return arg.name + "_height";
			}else {
				return arg.name + "_width";
			}
		}
		 else {
			return arg.name;
		}
	}
	private String getFunctionName() {
		return fileName;
	}
	public List<String> writeBody(ArrayList<Stmt> body , List<String> tokens ,
			Map<String,Type> environment) {
		this.environment = environment;
		writeThreadIndex(tokens);
		writeThreadGuard();
		for (Stmt statement : body) {
			write(statement,tokens);
		}
		return tokens;
	}
	private void writeThreadGuard() {
		// TODO Auto-generated method stub

	}
	private void writeThreadIndex(List<String> tokens) {
		//the 1D index
		int alias = 0;
		while (module.isArgument(indexName1D)) {
			indexName1D = "i"+Integer.toString(alias);
		}
		alias = 0;
		while (module.isArgument(indexName2D)) {
			indexName2D = "j"+Integer.toString(alias);
		}
		if (module.category != Category.GPUEXPLICITNONNESTED) { //simply write this since it works
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
			String formula = "threadIdx.x + ( blockDim.x * ( ( gridDim.x * blockIdx.y ) + blockIdx.x) ) ;";
			String[] parts = formula.split("\\s+");
			tokens.add("int");
			tokens.add(index2D());
			tokens.add("=");
			for (String part : parts) {
				tokens.add(part);
			}
//			tokens.add("int");
//			tokens.add(index2D());
//			tokens.add("=");
//			tokens.add("blockIdx.x");
//			tokens.add("*");
//			tokens.add("blockDim.x");
//			tokens.add("+");
//			tokens.add("threadIdx.x");
//			tokens.add(";");
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
			if (module.isArgument(((Expr.Variable)val).getName())) {
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
		//consider the expressions of the loop...
		//the kernel runner should take care of the thread running...
		for (Stmt stmt : statement.getBody()) {
			write(stmt,tokens);
		}

	}
	private String index1D() {
		return indexName1D;
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
			String lengthName = name+"_length"; //TODO fix me
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
		Expr src = indexOf.getSource();
		if (indexOf.getSource() instanceof Expr.Variable) {
			Expr indexVar = indexOf.getIndex();
			//indexVar is an instance of [int]
			//source expression must be of type...
			if (src instanceof Expr.Variable) {
				write1DIndexOf(tokens, (Expr.Variable) src, indexVar);
			}
			else {
				InternalFailure.internalFailure("Can only perform indexof on list", fileName, indexOf);
			}
		}else if (src instanceof Expr.IndexOf) {
			write2DIndexOf(tokens, indexOf);
		}else {
			InternalFailure.internalFailure("Expected source type to be of type list", fileName, indexOf.getSource());
		}
	}
	/**
	 *
	 * @param tokens
	 * @param indexOf
	 * @param outer
	 */
	private void write2DIndexOf(List<String> tokens, Expr.IndexOf indexOf) {
		Expr indexSrc = indexOf.getSource();
		Expr outerIndex = indexOf.getIndex();
		if (indexSrc instanceof Expr.IndexOf) {
			//the source is an index!
			Expr innerSrc = ((Expr.IndexOf) indexOf).getSource();
			//check if this indeed a nested indexof operation
			if (innerSrc instanceof Expr.IndexOf) {
				Expr innerInnerSrc = ((IndexOf) innerSrc).getSource();
				Expr innerIndex = ((Expr.IndexOf) innerSrc).getIndex();
				//the finally-indexed value must be a variable
				if (innerInnerSrc instanceof Expr.Variable) {
					Expr.Variable variable = (Variable) innerInnerSrc;
					//now have the inner src which is a variable
					//time to check out the indices
					if (outerIndex instanceof Expr.Variable && innerIndex
							instanceof Expr.Variable) {
						String innerName = ((Expr.Variable) outerIndex).getName();
						String outerName = ((Expr.Variable) innerIndex).getName();
						if (outerName.equals(module.getOuterIndex().getName())&&
								innerName.equals(module.getInnerIndex().getName())){
							//cleared for writing a 2d array access
							//first write the variable
							tokens.add(variable.getName());
							tokens.add("[");
							tokens.add(index2D());
							tokens.add("]");
						}
						else {
							InternalFailure.internalFailure("Writing non-loop index not implemented", fileName, indexOf);
						}
					}else {
						InternalFailure.internalFailure("IndexOf indices must both be nested loop indices", fileName, indexOf);
					}
				}else {
					InternalFailure.internalFailure("For 2D IndexOf, inner src must be variable", fileName, indexOf);
				}
			}
		}else {
			//fail here
			InternalFailure.internalFailure("For 2D IndexOf, outer src must be variable", fileName, indexOf);
		}
//			if ((innerIndex.getName().equals(module.getInnerIndex().getName())) {
//				write(indexSrc,tokens); //write the index source
//				//TODO compare indices here
//				tokens.add("[");
//				tokens.add(indexName2D);
//				tokens.add("]");
//			}else {
//				//TODO different behaviour here
//			}
//		}else {
//			//TODO there is an issue if this happens
//		}
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
				if (((Expr.Variable) indexVar).getName().equals(module.getOuterIndex().getName())) {
					if (module.category==Category.GPUEXPLICITNESTED) InternalFailure.internalFailure("Expected to index 1D list", fileName, indexVar);
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