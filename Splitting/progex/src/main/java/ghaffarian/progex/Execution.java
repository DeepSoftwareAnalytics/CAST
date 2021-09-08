/*** In The Name of Allah ***/
package ghaffarian.progex;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import ghaffarian.progex.graphs.ast.ASTBuilder;
import ghaffarian.progex.graphs.ast.AbstractSyntaxTree;
import ghaffarian.progex.graphs.cfg.CFGBuilder;
import ghaffarian.progex.graphs.cfg.ControlFlowGraph;
import ghaffarian.progex.graphs.cfg.ICFGBuilder;
import ghaffarian.progex.graphs.pdg.PDGBuilder;
import ghaffarian.progex.graphs.pdg.ProgramDependeceGraph;
import ghaffarian.progex.utils.FileUtils;
import ghaffarian.progex.utils.SystemUtils;
import ghaffarian.nanologger.Logger;
import ghaffarian.progex.java.JavaClass;
import ghaffarian.progex.java.JavaClassExtractor;
import java.util.List;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
/**
 * A class which holds program execution options.
 * These options determine the execution behavior of the program.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class Execution {
	
	private final ArrayList<Analysis> analysisTypes;
	private final ArrayList<String> inputPaths;
    private boolean debugMode;
	private String outputDir;
	private Languages lang;
	private NodeLevels nodeLevel;
	private Formats format;
	
	
	public Execution() {
        debugMode = false;
		analysisTypes = new ArrayList<>();
		inputPaths = new ArrayList<>();
		lang = Languages.JAVA;
		nodeLevel = NodeLevels.statement;
		format = Formats.DOT;
		outputDir = System.getProperty("user.dir");
		if (!outputDir.endsWith(File.separator))
			outputDir += File.separator;
	}
	
	/**
	 * Enumeration of different execution options.
	 */
	public enum Analysis {
		// analysis types
		CFG			("CFG"),
		PDG			("PDG"),
		AST			("AST"),
		ICFG		("ICFG"),
		SRC_INFO 	("INFO"),
		NODE_LEVEL  ("NODE_LEVEL");
		
		private Analysis(String str) {
			type = str;
		}
		@Override
		public String toString() {
			return type;
		}
		public final String type;
	}
	
	/**
	 * Enumeration of different supported languages.
	 */
	public enum Languages {
		C		("C", ".c"),
		JAVA	    ("Java", ".java"),
		PYTHON	("Python", ".py");
		
		private Languages(String str, String suffix) {
			name = str;
			this.suffix = suffix;
		}
		@Override
		public String toString() {
			return name;
		}
		public final String name;
		public final String suffix;
	}
	/**
	 * Enumeration of different supported Node Levels.
	 */
	public enum NodeLevels {
		statement, block
	}
	/**
	 * Enumeration of different supported output formats.
	 */
	public enum Formats {
		DOT, GML, JSON
	}
	
	
	/*=======================================================*/
	
	
	public void addAnalysisOption(Analysis opt) {
		analysisTypes.add(opt);
	}
	
	public void addInputPath(String path) {
		inputPaths.add(path);
	}
	
	public void setLanguage(Languages lang) {
		this.lang = lang;
	}
    
    public void setDebugMode(boolean isDebug) {
        debugMode = isDebug;
    }
    
    public void setNodeLevel(NodeLevels node_level) {
    	nodeLevel = node_level;
    }
	
	public void setOutputFormat(Formats fmt) {
		format = fmt;
	}
	
	public boolean setOutputDirectory(String outPath) {
        if (!outPath.endsWith(File.separator))
            outPath += File.separator;
		File outDir = new File(outPath);
        outDir.mkdirs();
		if (outDir.exists()) {
			if (outDir.canWrite() && outDir.isDirectory()) {
				outputDir = outPath;
				return true;
			}
		}
		return false;
	}
	
	@Override
	public String toString() {
		StringBuilder str = new StringBuilder();
		str.append("PROGEX execution config:");
		str.append("\n  Language = ").append(lang);
		str.append("\n  node level = ").append(nodeLevel);
		str.append("\n  Output format = ").append(format);
		str.append("\n  Output directory = ").append(outputDir);
		str.append("\n  Analysis types = ").append(Arrays.toString(analysisTypes.toArray()));
		str.append("\n  Input paths = \n");
		for (String path: inputPaths)
			str.append("        ").append(path).append('\n');
		return str.toString();
	}
	
	
	 public static void appendToFile(Exception e,String srcFile) {
		      try {
		         FileWriter fstream = new FileWriter("exception.txt", true);
		         BufferedWriter out = new BufferedWriter(fstream);
		         out.write(srcFile);
		         PrintWriter pWriter = new PrintWriter(out, true);
		         e.printStackTrace(pWriter);
		      }
		      catch (Exception ie) {
		         throw new RuntimeException("Could not write Exception to file", ie);
		      }
		   }
		
	/**
	 * Execute the PROGEX program with the given options.
	 */
	public void execute() {
		if (inputPaths.isEmpty()) {
			Logger.info("No input path provided!\nAbort.");
			System.exit(0);
		}
		if (analysisTypes.isEmpty()) {
			Logger.info("No analysis type provided!\nAbort.");
			System.exit(0);
		}
		
		Logger.info(toString());
		
		// 1. Extract source files from input-paths, based on selected language
		String[] paths = inputPaths.toArray(new String[inputPaths.size()]);
		String[] filePaths = new String[0];
		if (paths.length > 0)
			filePaths = FileUtils.listFilesWithSuffix(paths, lang.suffix);
		Logger.info("\n# " + lang.name + " source files = " + filePaths.length + "\n");
		
		// Check language
		if (!lang.equals(Languages.JAVA)) {
			Logger.info("Analysis of " + lang.name + " programs is not yet supported!");
			Logger.info("Abort.");
			System.exit(0);
		}

		// 2. For each analysis type, do the analysis and output results
		for (Analysis analysis: analysisTypes) {
			
			Logger.debug("\nMemory Status");
			Logger.debug("=============");
			Logger.debug(SystemUtils.getMemoryStats());
			int error_count = 0;
			switch (analysis.type) {
				//
				case "AST":
					Logger.info("\nAbstract Syntax Analysis");
					Logger.info("========================");
					Logger.debug("START: " + Logger.time() + '\n');
					error_count = 0;
					for (String srcFile : filePaths) {
						try {
							String[] file_paths = srcFile.split("/");
							String file_name = file_paths[file_paths.length - 1];
//							PrintStream out_s = 
//								new PrintStream(new FileOutputStream(
//									outputDir + file_name + ".ast.err.log"));
							PrintStream out_s = 
									new PrintStream(new FileOutputStream(
										file_name + ".ast.err.log"));
							System.setErr(out_s);
                            AbstractSyntaxTree ast = ASTBuilder.build(lang.name, srcFile, nodeLevel.toString());
							ast.export(format.toString(), outputDir);
						} catch (Exception ex) {
							// Logger.error(ex);
							ex.printStackTrace();
						}
						
//						try {
//							
//                            AbstractSyntaxTree ast = ASTBuilder.build(lang.name, srcFile, nodeLevel.toString());
//							ast.export(format.toString(), outputDir);
//						} catch (Exception ex) {
//							error_count++;
////							Logger.error(ex);
//							appendToFile(ex,srcFile);
////					         FileWriter fstream = new FileWriter("exception.txt", true);
////					         BufferedWriter out = new BufferedWriter(fstream);
////					         PrintWriter pWriter = new PrintWriter(out, true);
////					         e.printStackTrace(pWriter);
//						}
					}
//					Logger.info("number of error files: " + error_count);
					break;
				//
				case "CFG":
					Logger.info("\nControl-Flow Analysis");
					Logger.info("=====================");
					Logger.debug("START: " + Logger.time() + '\n');
					error_count = 0;
					for (String srcFile : filePaths) {
						try {
							String[] file_paths = srcFile.split("/");
							String file_name = file_paths[file_paths.length - 1];
							PrintStream out_s = 
								new PrintStream(new FileOutputStream(
									outputDir + file_name + ".cfg.err.log"));
//							PrintStream out_s = 
//									new PrintStream(new FileOutputStream(
//										file_name + ".cfg.err.log"));
							System.setErr(out_s);
							ControlFlowGraph cfg = CFGBuilder.build(lang.name, srcFile);
							cfg.export(format.toString(), outputDir);
						} catch (Exception ex) {
							// Logger.error(ex);
							ex.printStackTrace();
						}
//						try {//nodeLevel
//							ControlFlowGraph cfg = CFGBuilder.build(lang.name, srcFile);
//							cfg.nodeLevel = nodeLevel.toString();
//							cfg.export(format.toString(), outputDir);
//						} catch (Exception e) {
//							error_count++;
//							
//							appendToFile(e,srcFile );
////							Logger.error(ex);
////					         FileWriter fstream = new FileWriter("exception.txt", true);
////					         BufferedWriter out = new BufferedWriter(fstream);
////					         PrintWriter pWriter = new PrintWriter(out, true);
////					         e.printStackTrace(pWriter);
//						}
					}
//					Logger.info("number of error files: " + error_count);
					break;
				//
				case "ICFG":
					Logger.info("\nInterprocedural Control-Flow Analysis");
					Logger.info("=====================================");
					Logger.debug("START: " + Logger.time() + '\n');
					error_count = 0;
					try {
						ControlFlowGraph icfg = ICFGBuilder.buildForAll(lang.name, filePaths);
						icfg.export(format.toString(), outputDir);
					} catch (IOException ex) {
						Logger.error(ex);
					}
					break;
				//
				case "PDG":
					Logger.info("\nProgram-Dependence Analysis");
					Logger.info("===========================");
					Logger.debug("START: " + Logger.time() + '\n');
					error_count = 0;
					try {
						for (ProgramDependeceGraph pdg: PDGBuilder.buildForAll(lang.name, filePaths)) {
							pdg.CDS.export(format.toString(), outputDir);
							pdg.DDS.export(format.toString(), outputDir);
                            if (debugMode) {
                                pdg.DDS.getCFG().export(format.toString(), outputDir);
                                pdg.DDS.printAllNodesUseDefs(Logger.Level.DEBUG);
                            }
						}
					} catch (IOException ex) {
						Logger.error(ex);
					}
					break;
				//
				case "INFO":
					Logger.info("\nCode Information Analysis");
					Logger.info("=========================");
					Logger.debug("START: " + Logger.time() + '\n');
					for (String srcFile : filePaths)
						analyzeInfo(lang.name, srcFile);
					break;
				//
				default:
					Logger.info("\n\'" + analysis.type + "\' analysis is not supported!\n");
			}
			Logger.debug("\nFINISH: " + Logger.time());
		}
		//
		Logger.debug("\nMemory Status");
		Logger.debug("=============");
		Logger.debug(SystemUtils.getMemoryStats());
	}
    
	private void analyzeInfo(String lang, String srcFilePath) {
		switch (lang.toLowerCase()) {
			case "c":
				return;
			//
			case "java":
				try {
					Logger.info("\n========================================\n");
					Logger.info("FILE: " + srcFilePath);
					// first extract class info
					List<JavaClass> classInfoList = JavaClassExtractor.extractInfo(srcFilePath);
					for (JavaClass classInfo : classInfoList)
						Logger.info("\n" + classInfo);
					// then extract imports info
					if (classInfoList.size() > 0) {
						Logger.info("\n- - - - - - - - - - - - - - - - - - - - -");
						String[] imports = classInfoList.get(0).IMPORTS;
						for (JavaClass importInfo : JavaClassExtractor.extractImportsInfo(imports)) 
							Logger.info("\n" + importInfo);
					}
				} catch (IOException ex) {
					Logger.error(ex);
				}
				return;
			//
			case "python":
				return;
		}
	}
}
