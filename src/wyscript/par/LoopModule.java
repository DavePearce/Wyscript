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

import wyscript.lang.Expr;
import wyscript.lang.Stmt;
import wyscript.lang.Type;
import wyscript.lang.Expr.IndexOf;
import wyscript.lang.Expr.Variable;
import wyscript.util.SyntaxError.InternalFailure;

public class LoopModule {
	private static final String NVCC_COMMAND = "/opt/cuda/bin/nvcc ";
	private Stmt.ParFor loop;

	private List<String> tokens = new ArrayList<String>();
	private List<String> parameters = new ArrayList<String>();
	private Map<String , Type> environment; //passed to kernel writer at runtime
	private Set<String> nonParameterVars = new HashSet<String>();
	private Map<String,String> lengthMap = new HashMap<String,String>();

	private String indexName = "i";
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
	public LoopModule(String filename , Map<String , Type> environment , Stmt.ParFor loop){
		this.environment = environment;
		this.fileName = filename;
		loop.getBody();
		this.loop = loop;
		generateFunctionParameters(loop.getBody());
		writeFunctionDeclaration(tokens);
		tokens.add("{");
		//writeBody(body,tokens);
		tokens.add("}");
		try {
			saveAndCompileKernel(fileName+".cu");
		} catch (IOException e) {
			InternalFailure.internalFailure(
					"Could not save kernel file. Got error: "+e.getMessage(), fileName, loop);
		}
	}
	public void saveAndCompileKernel(String name) throws IOException {
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
		//first exclude the loop indices where necessary.
		//also scan the expression of each
		if (loop.depth == 3) {
			nonParameterVars.add(loop.indexZ.getName());
			scanExpr(loop.srcZ);
		}
		if (loop.depth >= 2) {
			nonParameterVars.add(loop.indexY.getName());
			scanExpr(loop.srcY);
		}
		if (loop.depth >= 1) {
			nonParameterVars.add(loop.indexX.getName());
			scanExpr(loop.srcX);
		}
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
				dealiaseIndex(name);
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
	private void dealiaseIndex(String name) {
		int dealiaser = 0;
		if (name.equals(indexName)) {
			do {
				indexName = "i_" + dealiaser;
				dealiaser++;
			} while (environment.containsKey(indexName)||
					parameters.contains(indexName));
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
		}else if (expr instanceof Expr.Unary) {
			scanExpr(((Expr.Unary) expr).getExpr());
		}
		else {
			//should not have to worry, this expr won't need params
		}
	}
	/**
	 * Writes the actual kernel's function declaration including name and arguments
	 *
	 * @requires The list of parameters to be written is initialised
	 * @ensures The function declaration is written with the required parameters
	 */
	private List<String> writeFunctionDeclaration(List<String> tokens) {
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
				writeListToDecl(tokens, name, type, list);
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
		return tokens;
	}
	/**
	 * Writes a list to the function declaration
	 * @param tokens
	 * @param name
	 * @param type
	 * @param list
	 */
	private void writeListToDecl(List<String> tokens, String name, Type type,
			Type.List list) {
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
		}else if (list.getElement() instanceof Type.List) {
			Type.List listType = (Type.List) list.getElement();
			Type elementType = listType.getElement();
			if (elementType instanceof Type.Int) {
				//now add int* arg, plus width and height
				tokens.add("int*");
				tokens.add(name);
				tokens.add(",");
				tokens.add("int*");
				tokens.add(name+"_width");
				tokens.add(",");
				tokens.add("int*");
				tokens.add(name+"_height");
			} else {
				InternalFailure.internalFailure("List of list should contain type Int only", fileName, list);
			}
		}
		else {
			InternalFailure.internalFailure("List type should be int for kernel conversion", fileName, list);
		}
	}
	public String getFuncName() {
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
		dealiaseIndex(lhs.getName());
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
			if (!srcVar.getName().equals(loop.indexX.getName())) {
				//make sure this expression variable doesnt share name with other indices.
				if (loop.indexY != null && loop.indexY.getName().equals(srcVar.getName())) return;
				if (loop.indexZ != null && loop.indexZ.getName().equals(srcVar.getName())) return;
				//parameters.add(srcVar.getName());
				scanVariableParam(srcVar);
			}
		}else {
			InternalFailure.internalFailure("Expression in index was not " +
					"variable which cannot match loop index", fileName, indexOf);
		}
	}
}
