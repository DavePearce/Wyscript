package wyscript.par;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import wyscript.lang.Expr;
import wyscript.lang.Expr.*;
import wyscript.lang.Stmt;
import wyscript.lang.Stmt.For;
import wyscript.lang.Type;
import wyscript.lang.WyscriptFile;
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
 * <li>assignment</li>
 * <li>variable declarations</li>
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

	private int begin;
	private int end;
	private int increment;

	private List<String> tokens = new ArrayList<String>();
	private List<String> parameters = new ArrayList<String>();

	private Map<String , Type> environment; //passed to kernel writer at runtime
	//this is the mapping from array to array length
	private Map<String,String> lengthFunction = new HashMap<String,String>();
	public Map<String, Type> getEnvironment() {
		return environment;
	}
	private String indexName = "i"; //TODO assign me an alias if name taken
	private String fileName;
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
		generateFunctionParameters();
		writeFunctionDeclaration();
		tokens.add("{");
		convertBody(body);
		tokens.add("}");
	}
	public void saveAndCompileKernel(String name) throws IOException {
		// first save the token list to file
		File file = new File(name);
		FileWriter  writer = new FileWriter(file);
		for (String token : tokens) {
			writer.write(token);
		}
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
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists()) {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            NVCC_COMMAND + " " + modelString + " -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;

        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
            new String(toByteArray(process.getErrorStream()));
        String outputMessage =
            new String(toByteArray(process.getInputStream()));
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
        	InternalFailure.internalFailure("Failed to compile .ptx file for " +
        			"kernel. \nnvcc returned "+exitValue, cuFileName, loop);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;

    }

    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream)
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

	/**
	 * This method generates a string of function parameters and analyses the
	 * loop body for those assignment statements which require parameters to be
	 * written to kernel.
	 *
	 * @requires loop != null and loop contains no illegal statements
	 * @ensures that all parameters necessary for a Cuda kernel are stored.
	 */
	private void generateFunctionParameters() {
		//scan the loop body, determine what must be added as parameter
		for (Stmt statement : loop.getBody()) {
			//an assign statement implies a mutable change
			if (statement instanceof Stmt.Assign) {
				//this is an assignment, must check the LHS and see where if comes from
				Stmt.Assign assign = (Stmt.Assign) statement;
				//check left hand side first
				if (assign.getLhs() instanceof Expr.IndexOf) {
					addIndexOfParam((Expr.IndexOf)assign.getLhs());
				}else if (assign.getLhs() instanceof Expr.Variable) {
					addVariableParam((Expr.Variable)assign.getLhs());
				}else {
					InternalFailure.internalFailure("Unknown LVal encountered."
							, fileName, assign.getLhs());
				}
				//now
				if (assign.getRhs() instanceof Expr.IndexOf) {
					addIndexOfParam((Expr.IndexOf)assign.getRhs());
				}else {
					InternalFailure.internalFailure("Unknown LVal encountered."
							, fileName, assign.getLhs());
				}

				//write(left);
			}
			else if (statement instanceof Stmt.VariableDeclaration) {
				String name = ((Stmt.VariableDeclaration)statement).getName();
				int dealiaser = 0;
				if (name.equals(indexName)) {
					//this means the index name has to be mangled
					//TODO verify the mangler here
					do {
						indexName = "i_" + dealiaser;
						dealiaser++;
					} while (environment.containsKey(indexName)||
							parameters.contains(indexName));
				}
			}
		}
	}
	/**
	 * Writes the actual kernel's function declaration including name and arguments
	 *
	 * @requires The list of parameters to be written is initialised
	 * @ensures The function declaration is written with the required parameters
	 */
	private void writeFunctionDeclaration() {
		tokens.add("__global__");
		tokens.add("void");
		tokens.add(getFuncName());
		tokens.add("(");
		for (int i = 0; i < parameters.size() ; i++) {
			//TODO WARNING potential off-by-one error
			if (i>=1 && i < parameters.size()) {
				tokens.add(",");
			}
			String name = parameters.get(i);
			//now work out the type of each parameters
			Type type = environment.get(name);
			if (type == null) {
				InternalFailure.internalFailure("Cannot retieve type for name "+name
						, fileName, type);
			}
			//write an array pointer, and also give a parameter for length
			if (type instanceof Type.List) {
				Type.List list = (Type.List) type;
				if (list.getElement() instanceof Type.Int) {
					tokens.add("int*");
					tokens.add(name);
					//note that the length is added to the list parameter
					tokens.add(",");
					tokens.add("int");
					//qualify length of array with '_length'
					tokens.add(name + "_length");
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
	private String getFuncName() {
		//TODO either remove me or implement me
		return fileName;
	}
	/**
	 * Add a single variable parameter to parameter list
	 * @param lhs
	 */
	private void addVariableParam(Variable lhs) {
		parameters.add(lhs.getName());
	}
	/**
	 * Add an indexOf operation as parameter. indexOf should be a flat access
	 * to an int value, and this will be checked.
	 * @param indexOf
	 *
	 * @requires indexOf != null and its source to be of Wyscript [int] type
	 * @ensures This parameter added to kernel parameter list
	 */
	private void addIndexOfParam(IndexOf indexOf) {
		Expr expression = indexOf.getSource();
		if (expression instanceof Expr.Variable) {
			Expr.Variable srcVar = (Expr.Variable)expression;
			if (!srcVar.getName().equals(loop.getIndex().getName())) {
				parameters.add(srcVar.getName());
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
	private void convertBody(ArrayList<Stmt> body) {
		writeThreadIndex();
		for (Stmt statement : body) {
			write(statement);
		}
	}
	private void writeThreadIndex() {
		tokens.add("int");
		tokens.add(indexName);
		tokens.add("=");
		tokens.add("blockIdx.x * blockDim.x + threadIdx.x");
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
	 * @requires
	 * @ensures
	 */
	private void write(Expr expression) {
		//TODO What is wrong here? Are there subtypes of expression who overload this method and bypass it completely.
		//if (expression instanceof Expr.)
		if (expression instanceof Expr.LVal) {
			write((Expr.LVal)expression);
		}else{
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
		if (val instanceof Expr.Variable) {
			//if this is a parameter, have to dereference the pointer
			if (parameters.contains(((Expr.Variable)val).getName())) {
					tokens.add("*");
			}
		}
		if (val instanceof Expr.Variable) {
			//write a
			Expr.Variable variable = (Expr.Variable) val;
			//simply add the variable name
			tokens.add(variable.getName());
		}else if (val instanceof Expr.IndexOf) {
			writeIndexOf(val);
		}

	}
	/**
	 * Checks whether an indexOf Expr.LVal left-hand expression matched the
	 * correct type for kernel conversion. Then writes it to the token list.
	 * @param val
	 */
	private void writeIndexOf(Expr.LVal val) {
		Expr.IndexOf indexOf = (Expr.IndexOf) val;
		if (indexOf.getSource() instanceof Expr.Variable) {
			Expr.Variable indexVar = (Expr.Variable)indexOf.getSource();
			//indexVar is an instance of [int]
			//source expression must be of type...
			Type typeOfVar = environment.get(indexVar.getName());
			if (typeOfVar instanceof Type.List) {
				Type listType = ((Type.List)typeOfVar).getElement();
				if (listType instanceof Type.Int) {
					//the type is correct for a kernel, write it here
					tokens.add(indexVar.getName());
					tokens.add("["+indexName+"]");
				}else{
					InternalFailure.internalFailure("List type should be int for kernel conversion", null, indexVar);
				}
			}

		}else {
			InternalFailure.internalFailure("Expected source type to be of type list", null, indexOf.getSource());
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
		return null;
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
	public int getBegin() {
		return begin;
	}
	public int getEnd() {
		return end;
	}
	public int getIncrement() {
		return increment;
	}
}