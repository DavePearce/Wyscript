// This file is part of the WyScript Compiler (wysc).
//
// The WyScript Compiler is free software; you can redistribute
// it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// The WyScript Compiler is distributed in the hope that it
// will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public
// License along with the WyScript Compiler. If not, see
// <http://www.gnu.org/licenses/>
//
// Copyright 2013, David James Pearce.

package wyscript;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import wyscript.io.*;
import wyscript.lang.WyscriptFile;
import wyscript.util.*;

public class Main {

	public static PrintStream errout;

	static {
		try {
			errout = new PrintStream(System.err, true, "UTF8");
		} catch (Exception e) {
			errout = System.err;
		}
	}

	private static enum Mode { interpret, js };

	public static boolean run(String[] args) {
		boolean verbose = false;
		int fileArgsBegin = 0;
		Mode mode = Mode.interpret;

		for (int i = 0; i != args.length; ++i) {
			if (args[i].startsWith("-")) {
				String arg = args[i];
				if (arg.equals("-help")) {
					usage();
					System.exit(0);
				} else if (arg.equals("-version")) {
					System.out.println("While Language Compiler (wysc)");
					System.exit(0);
				} else if (arg.equals("-verbose")) {
					verbose = true;
				} else if (arg.equals("-js")) {
					mode = Mode.js;
				} else {
					throw new RuntimeException("Unknown option: " + args[i]);
				}

				fileArgsBegin = i + 1;
			}
		}

		if (fileArgsBegin == args.length) {
			usage();
			return false;
		}

		try {
			String filename = args[fileArgsBegin];
			File srcFile = new File(filename);

			// First, lex and parse the source file
			Lexer lexer = new Lexer(srcFile.getPath());
			Parser parser = new Parser(srcFile.getPath(), lexer.scan());
			WyscriptFile ast = parser.read();

			// Second, we'd want to perform some kind of type checking here.
			// new TypeChecker().check(ast);

			// Third, we'd want to run the interpreter or compile the file.
			switch(mode) {
			case interpret:
				new Interpreter().run(ast);
				break;
			case js: {
				File jsFile = new File(filename.substring(0,filename.lastIndexOf('.')) + ".js");
				JavaScriptFileWriter jsfw = new JavaScriptFileWriter(jsFile);
				jsfw.write(ast);
				jsfw.close();
				break;
			}
			}

		} catch (SyntaxError e) {
			if (e.filename() != null) {
				e.outputSourceError(System.out);
			} else {
				System.err.println("syntax error (" + e.getMessage() + ").");
			}

			if (verbose) {
				e.printStackTrace(errout);
			}

			return false;
		} catch (Exception e) {
			errout.println("Error: " + e.getMessage());
			if (verbose) {
				e.printStackTrace(errout);
			}
			return false;
		}

		return true;
	}

	public static void main(String[] args) throws Exception {
		run(args);
	}

	/**
	 * Print out information regarding command-line arguments
	 *
	 */
	public static void usage() {
		String[][] info = {
				{ "version", "Print version information" },
				{ "verbose",
						"Print detailed information on what the compiler is doing" } };

		System.out.println("usage: wyjs <options> <source-files>");
		System.out.println("Options:");

		// first, work out gap information
		int gap = 0;

		for (String[] p : info) {
			gap = Math.max(gap, p[0].length() + 5);
		}

		// now, print the information
		for (String[] p : info) {
			System.out.print("  -" + p[0]);
			int rest = gap - p[0].length();
			for (int i = 0; i != rest; ++i) {
				System.out.print(" ");
			}
			System.out.println(p[1]);
		}
	}
}
