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
import wyscript.lang.Expr.LVal;
import wyscript.lang.Expr.Unary;
import wyscript.lang.Expr.Variable;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.util.SyntacticElement;
import wyscript.util.SyntaxError.InternalFailure;


/**
 * The first instance of the KernelWriter will take an ordinary for-loop and convert it
 * to Cuda code. There are a number of limitations on what loops can be written. The first goal
 * is simple loops with <i>int</i> and <i>[int]</i> types, with no nested loops and simple
 * conditionals. The kernel writer will compile the cuda code and throw an exception if this operation
 * is not successful.
 *
 * Only certain types of parallel for (parFor) loops can be parallelised by this component.
 * So far they include loops with statement bodies with <i>only the following</i> statement types
 * <ul>
 * <li>assignments (int and [int])</li>
 * <li>variable declarations (int and [int])</li>
 * <li>if-else statements</li>
 * <li></li>
 * </ul>
 * Furthermore, expression types are limited to
 * <ul>
 * <li>Simple unary and binary operations</li>
 * <li>Index operations</li>
 * </ul>
 * @author Mate Antunovic
 *
 */
public class KernelWriter {
	private static final String NVCC_COMMAND = "/opt/cuda/bin/nvcc ";
	private ArrayList<Stmt> body;
	private Stmt.ParFor loop;

	private List<String> tokens = new ArrayList<String>();
	private List<String> parameters = new ArrayList<String>();
	private Map<String , Type> environment; //passed to kernel writer at runtime
	private Set<String> nonParameterVars = new HashSet<String>();
	private Map<String,String> lengthMap = new HashMap<String,String>();
	private String indexName = "i";
	private String fileName;
	private String ptxFileName;
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
	public KernelWriter(String filename , Map<String , Type> environment , Stmt.ParFor loop){
		this.environment = environment;
		this.fileName = filename;
		this.body = loop.getBody();
		this.loop = loop;
		generateFunctionParameters(loop.getBody());
		writeFunctionDeclaration();
		tokens.add("{");
		writeBody(body);
		tokens.add("}");
		try {
			saveAndCompileKernel(fileName+".cu");
		} catch (IOException e) {
			InternalFailure.internalFailure(
					"Could not save kernel file. Got error: "+e.getMessage(), fileName, loop);
		}
	}
	public void saveAndCompileKernel(String name) throws IOException {
		// first save the token list to file
		File file = new File(name);
		//System.out.println("Wrote file to "+file.getAbsolutePath());
		FileWriter  writer = new FileWriter(file);
		//System.out.println("Compiling in "+file.getAbsolutePath());

		for (String token : tokens) {
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
    private static String convertStreamToString(java.io.InputStream is) {
        java.util.Scanner s = new java.util.Scanner(is).useDelimiter("\\A");
        return s.hasNext() ? s.next() : "";
    }
    /**
	 * This method generates a string of function parameters and analyses the
	 * loop body for those assignment statements which require parameters to be
	 * written to kernel.
	 *
	 * @requires loop != null and loop contains no illegal statements
	 * @ensures that all parameters necessary for a Cuda kernel are stored.
	 */
	private void generateFunctionParameters(Collection<Stmt> body) {
		//scan the loop body, determine what must be added as parameter
		//first exclude the loop index
		nonParameterVars.add(loop.getIndex().getName());
		for (Stmt statement : body) {
		//check for mutabilities in assignment
			if (statement instanceof Stmt.Assign) {
				Stmt.Assign assign = (Stmt.Assign) statement;
				scanAssign(assign);
			}
		//ensure index variable not shadowed
			else if (statement instanceof Stmt.VariableDeclaration) {
				Stmt.VariableDeclaration vardec = (Stmt.VariableDeclaration)statement;
				String name = ((Stmt.VariableDeclaration)statement).getName();
				nonParameterVars.add(name);
				int dealiaser = 0;
				if (name.equals(indexName)) {
					do {
						indexName = "i_" + dealiaser;
						dealiaser++;
					} while (environment.containsKey(indexName)||
							parameters.contains(indexName));
				}
				scanExpr(vardec.getExpr());
			}else if (statement instanceof Stmt.IfElse) {
				Stmt.IfElse ifelse = (Stmt.IfElse) statement;
				scanExpr(ifelse.getCondition());
				generateFunctionParameters(ifelse.getTrueBranch());
				generateFunctionParameters(ifelse.getFalseBranch());
			}else {
				InternalFailure.internalFailure("Encountered unexpected statement type "
			+statement.getClass(), fileName, statement);
			}
		}
	}
	/**
	 * Scans the lhs and rhs of the assign statement.
	 * @param assign
	 */
	private void scanAssign(Stmt.Assign assign) {
		scanExpr(assign.getLhs());
		scanExpr(assign.getRhs());
	}
	/**
	 * If this expr represents an access to a variable or index, then it is
	 * added to the parameter list
	 * @param expr
	 */
	private void scanExpr(Expr expr) {
		if (expr instanceof Expr.Variable) {
			scanVariableParam((Variable) expr);
		}else if (expr instanceof Expr.Binary) {
			Expr.Binary binary = (Expr.Binary) expr;
			scanExpr(binary.getLhs());
			scanExpr(binary.getRhs());
		}else if (expr instanceof Expr.IndexOf) {
			scanIndexOf((Expr.IndexOf)expr);
		}else {
			//should not have to worry, this expr won't need params
		}
	}
	/**
	 * Writes the actual kernel's function declaration including name and arguments
	 *
	 * @requires The list of parameters to be written is initialised
	 * @ensures The function declaration is written with the required parameters
	 */
	private void writeFunctionDeclaration() {
		tokens.add("extern");
		tokens.add("\"C\"");
		tokens.add("__global__");
		tokens.add("void");
		tokens.add(getFuncName());
		tokens.add("(");
		for (int i = 0; i < parameters.size() ; i++) {
			if (i>=1 && i < parameters.size()) {
				tokens.add(",");
			}
			String name = parameters.get(i);
			//now work out the type of each parameters
			Type type = environment.get(name);
			if (type == null) {
				InternalFailure.internalFailure("Cannot retieve type for name "+name
						, fileName, new Type.Void());
			}
			//write an array pointer, and also give a parameter for length
			if (type instanceof Type.List) {
				Type.List list = (Type.List) type;
				if (list.getElement() instanceof Type.Int) {
					tokens.add("int*");
					tokens.add(name);
					//note that the length is added to the list parameter
					tokens.add(",");
					tokens.add("int*");
					//qualify length of array with '_length'
					String lengthName = name + "_length";
					if (parameters.contains(lengthName)) {
						//TODO make this fail-safe and de-alias the name
						InternalFailure.internalFailure("Parameter with name "+
					lengthName+" preventing addition of length parameter", fileName, type);
					}
					tokens.add(lengthName);
					lengthMap.put(name, lengthName);
				}else {
					InternalFailure.internalFailure("List type should be int for kernel conversion", null, list);
				}
			}else if (type instanceof Type.Int) {
				tokens.add("int*");
				tokens.add(name);
			}
			else {
				InternalFailure.internalFailure("Unknown parameter type encountered."
						, fileName, type);
			}
		}
		tokens.add(")");
	}
	public String getFuncName() {
		//TODO either remove me or implement me
		return fileName;
	}
	/**
	 * Add a single variable parameter to parameter list
	 * @param lhs
	 */
	private void scanVariableParam(Variable lhs) {
		if (!parameters.contains(lhs.getName()) &&
				!nonParameterVars.contains(lhs.getName())) parameters.add
				(lhs.getName());
	}
	/**
	 * Add an indexOf operation as parameter. indexOf should be a flat access
	 * to an int value, and this will be checked.
	 * @param indexOf
	 *
	 * @requires indexOf != null and its source to be of Wyscript [int] type
	 * @ensures This parameter added to kernel parameter list
	 */
	private void scanIndexOf(IndexOf indexOf) {
		Expr expression = indexOf.getSource();
		if (expression instanceof Expr.Variable) {
			Expr.Variable srcVar = (Expr.Variable)expression;
			if (!srcVar.getName().equals(loop.getIndex().getName())) {
				//parameters.add(srcVar.getName());
				scanVariableParam(srcVar);
			}
		}else {
			InternalFailure.internalFailure("Expression in index was not " +
					"variable which cannot match loop index", fileName, indexOf);
		}
	}
	/**
	 * Maps the body of the loop onto Cuda code
	 * @param body
	 *
	 * @requires body != null and every element of body is a legal statement
	 */
	private void writeBody(ArrayList<Stmt> body) {
		writeThreadIndex();
		writeGuardBegin();
		for (Stmt statement : body) {
			write(statement);
		}
		tokens.add("}"); //add end of guard
	}
	private void writeGuardBegin() {
		Expr src = loop.getSource();
		Expr.Binary srcBinary = (Expr.Binary) src;
		if (srcBinary.getOp().equals(Expr.BOp.RANGE)) {
			Expr low = srcBinary.getLhs();
			Expr.Unary high = (Unary) srcBinary.getRhs();
			tokens.add("if");
			tokens.add("(");
			tokens.add(indexName);
			tokens.add("<");
			write(high);
			tokens.add("&&");
			tokens.add(indexName);
			tokens.add(">=");
			write(low);
			tokens.add(")");
			tokens.add("{");
		}else {
			InternalFailure.internalFailure("Expected loop source to be range " +
					"operator", fileName, src);
		}
	}
	private void writeThreadIndex() {
		tokens.add("int");
		tokens.add(indexName);
		tokens.add("=");
		tokens.add("blockIdx.x");
		tokens.add("*");
		tokens.add("blockDim.x");
		tokens.add("+");
		tokens.add("threadIdx.x");
		tokens.add(";");
	}
	/**
	 * Convert a single statement to its appropriate kernel form. The statement must
	 * meet certain requirements of for conversion to Cuda code.
	 * @param statement
	 *
	 * @requires statement != null, statement is a legal and the parameters have been initialised
	 * @ensures A single statement of Cuda is written that correctly maps the Wyscript functionality
	 */
	private void write(Stmt statement) {
		// what happens here?
		if (statement instanceof Stmt.IfElse) {
			write((Stmt.IfElse)statement);
		}
		else if (statement instanceof Stmt.VariableDeclaration) {
			write((Stmt.VariableDeclaration) statement);
		}else if (statement instanceof Stmt.Assign) {
			write((Stmt.Assign)statement);
		}else {
			InternalFailure.internalFailure("Encountered syntactic element not " +
					"supported in parFor loop", fileName, statement);
		}
	}
	/**
	 * Writes an assignment statement to the kernel
	 * @param assign
	 *
	 * @requires
	 * @ensures
	 */
	private void write(Stmt.Assign assign) {
		Expr.LVal lhs = assign.getLhs();
		write(lhs);
		tokens.add("=");
		Expr rhs = assign.getRhs();
		write(rhs);
		tokens.add(";");
	}
	/**
	 * Writes a single expression to the kernel
	 * @param expression
	 *
	 * @requires expression is of an acceptable type and has appropriate parameters
	 * @ensures the Cuda form of the expression is written to the token list
	 */
	private void write(Expr expression) {
		if (expression instanceof Expr.LVal) {
			write((Expr.LVal)expression);
		}else if (expression instanceof Expr.Variable) {
			write((Expr.Variable) expression);
		}else if (expression instanceof Expr.ListConstructor) {
			write((Expr.ListConstructor) expression);
		}else if (expression instanceof Expr.Constant) {
			write((Expr.Constant) expression);
		}else if (expression instanceof Expr.IndexOf) {
			write((Expr.IndexOf)expression);
		}else if (expression instanceof Expr.Binary) {
			write((Expr.Binary) expression);
		}else if (expression instanceof Expr.Unary) {
			write((Expr.Unary)expression);
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
	private void write(Expr.LVal val) {
		tokens.add("(");
		if (val instanceof Expr.Variable) {
			//if this is a parameter, have to dereference the pointer
			if (parameters.contains(((Expr.Variable)val).getName())) {
					tokens.add("*");
			}
			//write a
			Expr.Variable variable = (Expr.Variable) val;
			//simply add the variable name
			tokens.add(variable.getName());
			tokens.add(")");
		}else if (val instanceof Expr.IndexOf) {
			write((Expr.IndexOf)val);
			tokens.add(")");
		}

	}
	private void write(Expr.Binary binary) {
		//TODO address the issue of precedence here
		tokens.add("(");
		write(binary.getLhs());
		tokens.add(")");
		writeOp(binary.getOp());
		tokens.add("(");
		write(binary.getRhs());
		tokens.add(")");
	}
	private void write(Expr.Unary unary) {
		//TODO fill me up so that I take length operations
		switch (unary.getOp()) {
		case LENGTHOF:
			writeLengthOf(unary.getExpr());
			break;
		case NEG:
			tokens.add("-");
			tokens.add("(");
			write(unary.getExpr());
			tokens.add(")");
			break;
		case NOT:
			tokens.add("!");
			tokens.add("(");
			write(unary.getExpr());
			tokens.add(")");
			break;
		default:
			InternalFailure.internalFailure("Unknown unary expression encountered"
					, fileName, unary);

		}
	}
	private void writeLengthOf(Expr expr) {
		if (expr instanceof Expr.Variable) {
			String name = ((Expr.Variable)expr).getName();
			tokens.add("(");
			tokens.add("*");
			String lengthName = lengthMap.get(name);
			tokens.add(lengthName);
			tokens.add(")");
		}else {
			//TODO Implement me
			InternalFailure.internalFailure("Writing length of this expression not implemented", fileName, expr);
		}
	}
	private void writeOp(BOp op) {
		tokens.add(op.toString());
	}
	/**
	 * Writes a single constant to token list. This constant may be an int only
	 * @param constant
	 *
	 * @requires the constant is an int
	 */
	private void write(Expr.Constant constant) {
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
	private void write(Expr.ListConstructor list) {
		InternalFailure.internalFailure("Writing list constructors not implemented"
				, fileName, list);
	}
	/**
	 * Write a single indexOf operation to token list
	 * @param expr
	 */
	private void write(Expr.IndexOf indexOf) {
		if (indexOf.getSource() instanceof Expr.Variable) {
			Expr.Variable srcVar = (Expr.Variable)indexOf.getSource();
			//indexVar is an instance of [int]
			//source expression must be of type...
			Type typeOfVar = environment.get(srcVar.getName());
			if (typeOfVar instanceof Type.List) {
				Type listType = ((Type.List)typeOfVar).getElement();
				if (listType instanceof Type.Int) {
					//the type is correct for a kernel, write it here
					tokens.add(srcVar.getName());
					tokens.add("["+indexName+"]");
				}else{
					InternalFailure.internalFailure("List type should be int for kernel conversion", fileName, srcVar);
				}
			}else {
				InternalFailure.internalFailure("Can only perform indexof on list", fileName, indexOf);
			}

		}else {
			InternalFailure.internalFailure("Expected source type to be of type list", fileName, indexOf.getSource());
		}
		if (indexOf.getIndex().equals(loop.getIndex())) { //TODO Potential issue with comparing indices

		}
	}
	/**
	 * Writes a classical conditional statement to the kernel
	 * @param statement
	 */
	private void write(Stmt.IfElse statement) {
		tokens.add("if");
		tokens.add("(");
		//the condition can only be simple equality, or a statement in boolean
		//logic
		Expr expression = statement.getCondition();
		write(expression);
		tokens.add(")");
		tokens.add("{");
		//branches may be empty
		for (Stmt s : statement.getTrueBranch()) {
			write(s); //write the single statement
		}
		tokens.add("}");
		tokens.add("else");
		tokens.add("{");
		for (Stmt s : statement.getFalseBranch()) {
			write(s); //write the single statement
		}
		tokens.add("}");
	}
	/**
	 * Writes a single variable declaration to the kernel.
	 * @param decl
	 */
	private void write(Stmt.VariableDeclaration decl) {
		Type type = decl.getType();
		if (type instanceof Type.Int) {
			tokens.add("int");
			tokens.add(decl.getName());
			tokens.add("=");
			//now write the expression
			write(decl.getExpr());
			tokens.add(";");
		}else {
			InternalFailure.internalFailure("Cannot write variable declaration for the given type",null,null);

		}
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
	/**
	 * Generates a Kernel Runner from the par-for loop
	 * @return
	 */
	public KernelRunner getRunner() {
		return new KernelRunner(this);
	}
	public List<String> getParameters() {
		return new ArrayList<String>(parameters);
	}
	public SyntacticElement getLoop() {
		return loop;
	}
	public Map<String, Type> getEnvironment() {
		return environment;
	}
}