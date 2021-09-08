/*** In The Name of Allah ***/
package ghaffarian.progex.java;

import ghaffarian.graphs.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.misc.Interval;
import org.antlr.v4.runtime.tree.ParseTree;
import ghaffarian.progex.graphs.cfg.CFEdge;
import ghaffarian.progex.graphs.cfg.CFNode;
import ghaffarian.progex.graphs.cfg.ControlFlowGraph;
import ghaffarian.progex.java.parser.JavaBaseVisitor;
import ghaffarian.progex.java.parser.JavaLexer;
import ghaffarian.progex.java.parser.JavaParser;
import ghaffarian.nanologger.Logger;
import org.antlr.v4.gui.TreeViewer;
/**
 * A Control Flow Graph (CFG) builder for Java programs.
 * A Java parser generated via ANTLRv4 is used for this purpose.
 * This implementation is based on ANTLRv4's Visitor pattern.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class JavaCFGBuilder {
	
	/**
	 * ‌Build and return the Control Flow Graph (CFG) for the given Java source file.
	 */
	public static ControlFlowGraph build(String javaFile) throws IOException {
		return build(new File(javaFile));
	}
	
	/**
	 * ‌Build and return the Control Flow Graph (CFG) for the given Java source file.
	 */
	public static ControlFlowGraph build(File javaFile) throws IOException {
		if (!javaFile.getName().endsWith(".java"))
			throw new IOException("Not a Java File!");
		InputStream inFile = new FileInputStream(javaFile);
		ANTLRInputStream input = new ANTLRInputStream(inFile);
		JavaLexer lexer = new JavaLexer(input);
		CommonTokenStream tokens = new CommonTokenStream(lexer);
		JavaParser parser = new JavaParser(tokens);
		ParseTree tree = parser.compilationUnit();
        //show AST in console
//        System.out.println(tree.toStringTree(parser));

        //show AST in GUI
//        TreeViewer viewr = new TreeViewer(Arrays.asList(
//                parser.getRuleNames()),tree);
//        viewr.open();
		return build(javaFile.getName(), tree, null, null);
	}
	
	/**
	 * ‌Build and return the Control Flow Graph (CFG) for the given Parse-Tree.
	 * The 'ctxProps' map includes contextual-properties for particular nodes 
	 * in the parse-tree, which can be used for linking this graph with other 
	 * graphs by using the same parse-tree and the same contextual-properties.
	 */
	public static ControlFlowGraph build(String javaFileName, ParseTree tree, 
			String propKey, Map<ParserRuleContext, Object> ctxProps) {
		ControlFlowGraph cfg = new ControlFlowGraph(javaFileName);
		ControlFlowVisitor visitor = new ControlFlowVisitor(cfg, propKey, ctxProps);
		visitor.visit(tree);
		return cfg;
	}
	
	/**
	 * Visitor-class which constructs the CFG by walking the parse-tree.
	 */
	private static class ControlFlowVisitor extends JavaBaseVisitor<Void> {
		
		private ControlFlowGraph cfg;
		private Deque<CFNode> preNodes;
		private Deque<CFEdge.Type> preEdges;
		private Deque<Block> loopBlocks;
		private List<Block> labeledBlocks;
		private Deque<Block> tryBlocks;
		private Queue<CFNode> casesQueue;
		private boolean dontPop;
		private String propKey;
		private Map<ParserRuleContext, Object> contexutalProperties;
		private Deque<String> classNames;
//		private String nodeLevel;

		public ControlFlowVisitor(ControlFlowGraph cfg, String propKey, Map<ParserRuleContext, Object> ctxProps) {
			preNodes = new ArrayDeque<>();
			preEdges = new ArrayDeque<>();
			loopBlocks = new ArrayDeque<>();
			labeledBlocks = new ArrayList<>();
			tryBlocks = new ArrayDeque<>();
			casesQueue = new ArrayDeque<>();
			classNames = new ArrayDeque<>();
			dontPop = false;
//			nodeLevel = node_level;
			this.cfg = cfg;
			//
			this.propKey = propKey;
			contexutalProperties = ctxProps;
		}

		/**
		 * Reset all data-structures and flags for visiting a new method declaration.
		 */
		private void init() {
			preNodes.clear();
			preEdges.clear();
			loopBlocks.clear();
			labeledBlocks.clear();
			tryBlocks.clear();
			dontPop = false;
		}
		//label_end_block
		public void label_end_block(CFNode endBlock)
		{
			boolean flag = dontPop;
			dontPop = false;
			CFNode lastNode = preNodes.pop();
			cfg.justRemoveVertex(lastNode );
			cfg.addVertex(lastNode);
			lastNode.setProperty("is_end_block",true);
			preNodes.push(lastNode);
//			cfg.addVertex(endelse);
			popAddPreEdgeTo(endBlock);
			dontPop = flag;
//			try {
//				boolean flag = dontPop;
//				dontPop = false;
//				CFNode lastNode = preNodes.pop();
//				cfg.justRemoveVertex(lastNode );
//				cfg.addVertex(lastNode);
//				lastNode.setProperty("is_end_block",true);
//				preNodes.push(lastNode);
////				cfg.addVertex(endelse);
//				popAddPreEdgeTo(endBlock);
//				dontPop = flag;
//			}
//		catch (Exception e) {
//				
//		}
				
		}
		
		// add start block
		public Void label_start_block(int next_line,String label) {
			CFNode startIf = new CFNode();
//			int privious_line = preNodes;
			startIf.setLineOfCode(-100*next_line+50 );
			startIf.setCode(label);
			addNodeAndPreEdge(startIf);
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(startIf);
			return null;
			}
		
		//get the end_block_line
		public int get_end_block_line() {
			CFNode lastNode = preNodes.pop();
			int Line = lastNode.getLineOfCode() + 1;
			preNodes.push(lastNode);
			Line = -100 * Line + 50 ;
			return Line;
		}
		/**
		 * Add contextual properties to the given node.
		 * This will first check to see if there is any property for the 
		 * given context, and if so, the property will be added to the node.
		 */
		private void addContextualProperty(CFNode node, ParserRuleContext ctx) {
			if (propKey != null && contexutalProperties != null) {
				Object prop = contexutalProperties.get(ctx);
				if (prop != null)
					node.setProperty(propKey, prop);
			}
		}
		
		@Override
		public Void visitPackageDeclaration(JavaParser.PackageDeclarationContext ctx) {
			// packageDeclaration :  annotation* 'package' qualifiedName ';'
			cfg.setPackage(ctx.qualifiedName().getText());
			return null;
		}

		@Override
		public Void visitClassDeclaration(JavaParser.ClassDeclarationContext ctx) {
			// classDeclaration 
			//   :  'class' Identifier typeParameters? 
			//      ('extends' typeType)? ('implements' typeList)? classBody
			classNames.push(ctx.Identifier().getText());
			visit(ctx.classBody());
			classNames.pop();
			return null;
		}

		@Override
		public Void visitEnumDeclaration(JavaParser.EnumDeclarationContext ctx) {
			// Just ignore enums for now ...
			return null;
		}
		
		@Override
		public Void visitInterfaceDeclaration(JavaParser.InterfaceDeclarationContext ctx) {
			// Just ignore interfaces for now ...
			return null;
		}
		
		@Override
		public Void visitClassBodyDeclaration(JavaParser.ClassBodyDeclarationContext ctx) {
			// classBodyDeclaration :  ';'  |  'static'? block  |  modifier* memberDeclaration
			if (ctx.block() != null) {
				init();
				//
				CFNode block = new CFNode();
				if (ctx.getChildCount() == 2 && ctx.getChild(0).getText().equals("static")) {
					block.setLineOfCode(ctx.getStart().getLine());
					block.setCode("static");
				} else {
					block.setLineOfCode(0);
					block.setCode("block");
				}
				addContextualProperty(block, ctx);
				cfg.addVertex(block);
				//
				block.setProperty("name", "static-block");
				block.setProperty("class", classNames.peek());
				cfg.addMethodEntry(block);
				//
				preNodes.push(block);
				preEdges.push(CFEdge.Type.EPSILON);
			}
			return visitChildren(ctx);
		}

		@Override
		public Void visitConstructorDeclaration(JavaParser.ConstructorDeclarationContext ctx) {
			// Identifier formalParameters ('throws' qualifiedNameList)?  constructorBody
			init();
			//
			CFNode entry = new CFNode();
			entry.setLineOfCode(ctx.getStart().getLine());
			entry.setCode(ctx.Identifier().getText() + ' ' + getOriginalCodeText(ctx.formalParameters()));
			addContextualProperty(entry, ctx);
			cfg.addVertex(entry);
			//
			entry.setProperty("name", ctx.Identifier().getText());
			entry.setProperty("class", classNames.peek());
			cfg.addMethodEntry(entry);
			//
			preNodes.push(entry);
			preEdges.push(CFEdge.Type.EPSILON);
			return visitChildren(ctx);
		}

		@Override
		public Void visitMethodDeclaration(JavaParser.MethodDeclarationContext ctx) {
			// methodDeclaration :
			//   (typeType|'void') Identifier formalParameters ('[' ']')*
			//     ('throws' qualifiedNameList)?  ( methodBody | ';' )
			init();
			//
			CFNode entry = new CFNode();
			entry.setLineOfCode(ctx.getStart().getLine());
			String retType = "void";
			if (ctx.typeType() != null)
				retType = ctx.typeType().getText();
			String args = getOriginalCodeText(ctx.formalParameters());
			entry.setCode(retType + " " + ctx.Identifier() + args);
			addContextualProperty(entry, ctx);
			cfg.addVertex(entry);
			//
			entry.setProperty("name", ctx.Identifier().getText());
			entry.setProperty("class", classNames.peek());
			entry.setProperty("type", retType);
			cfg.addMethodEntry(entry);
			//
			preNodes.push(entry);
			preEdges.push(CFEdge.Type.EPSILON);
			return visitChildren(ctx);
		}

		@Override
		public Void visitStatementExpression(JavaParser.StatementExpressionContext ctx) {
			// statementExpression ';'
			CFNode expr = new CFNode();
			expr.setLineOfCode(ctx.getStart().getLine());
			expr.setCode(getOriginalCodeText(ctx));
			//
			Logger.debug(expr.getLineOfCode() + ": " + expr.getCode());
			//
			boolean is_simple_stmt = true;
			expr.setProperty("is_simple_stmt", is_simple_stmt);
			addContextualProperty(expr, ctx);
			addNodeAndPreEdge(expr);
			//
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(expr);
			return null;
		}
		
		@Override
		public Void visitLocalVariableDeclaration(JavaParser.LocalVariableDeclarationContext ctx) {
			// localVariableDeclaration :  variableModifier* typeType variableDeclarators
			CFNode declr = new CFNode();
			declr.setLineOfCode(ctx.getStart().getLine());
			declr.setCode(getOriginalCodeText(ctx));
			declr.setProperty("is_simple_stmt",  true);
			
			addContextualProperty(declr, ctx);
			addNodeAndPreEdge(declr);
			//
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(declr);
			return null;
		}
		

		
		@Override
		public Void visitIfStatement(JavaParser.IfStatementContext ctx) {
			//start for
			int nextLine = ctx.getStart().getLine();
			label_start_block(nextLine,"[stratif]" );
			
			// 'if' parExpression statement ('else' statement)?
			CFNode ifNode = new CFNode();
			ifNode.setLineOfCode(ctx.getStart().getLine());
			ifNode.setCode("if " + getOriginalCodeText(ctx.parExpression()));
			addContextualProperty(ifNode, ctx);
			addNodeAndPreEdge(ifNode);
			//
			preEdges.push(CFEdge.Type.TRUE);
			preNodes.push(ifNode);
			//
			visit(ctx.statement(0));
			//
			CFNode endif = new CFNode();
		
//			endif.setLineOfCode(get_end_block_line());
			endif.setLineOfCode(-100*nextLine+25);
			endif.setCode("endif");
			addNodeAndPreEdge(endif);
			
//			CFNode endelse = new CFNode();
//			endelse.setLineOfCode(-4);
//			endelse.setCode("if");
			//
			if (ctx.statement().size() == 1) { // if without else
				cfg.addEdge(new Edge<>(ifNode, new CFEdge(CFEdge.Type.FALSE), endif));
//				cfg.addVertex(endelse);
			} else {  //  if with else
				preEdges.push(CFEdge.Type.FALSE);
				preNodes.push(ifNode);
				visit(ctx.statement(1));
				// Edit last statement in the block
				label_end_block(endif);
				
			}
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(endif);
			return null;
		}

		

		
		@Override
		public Void visitForStatement(JavaParser.ForStatementContext ctx) {
			//start for
//			CFNode startFor = new CFNode();
//			startFor.setLineOfCode(-1);
//			startFor.setCode("[startfor]");
//			addNodeAndPreEdge(startFor);
//			preEdges.push(CFEdge.Type.EPSILON);
//			preNodes.push(startFor);
			int nextLine = ctx.forControl().getStart().getLine();
			label_start_block( nextLine,"[startfor]");
			// 'for' '(' forControl ')' statement
			//  First, we should check type of for-loop ...
		
			if (ctx.forControl().enhancedForControl() != null) {
				// This is a for-each loop;
				//   enhancedForControl: 
				//     variableModifier* typeType variableDeclaratorId ':' expression
				CFNode forExpr = new CFNode();
				forExpr.setLineOfCode(ctx.forControl().getStart().getLine());
				forExpr.setCode("for (" + getOriginalCodeText(ctx.forControl()) + ")");
				addContextualProperty(forExpr, ctx.forControl().enhancedForControl());
				addNodeAndPreEdge(forExpr);
				//
				CFNode forEnd = new CFNode();
				forEnd.setLineOfCode(-100*nextLine+25);
				forEnd.setCode("[endfor]");
				cfg.addVertex(forEnd);
				cfg.addEdge(new Edge<>(forExpr, new CFEdge(CFEdge.Type.FALSE), forEnd));
				//
				preEdges.push(CFEdge.Type.TRUE);
				preNodes.push(forExpr);
				//
				loopBlocks.push(new Block(forExpr, forEnd));
				visit(ctx.statement());
				loopBlocks.pop();
				// add endfor
				CFNode lastNode = preNodes.pop();
				cfg.addEdge(new Edge<>(lastNode, new CFEdge(CFEdge.Type.ENDLOOP), forEnd));
				preNodes.push(lastNode);
				
				popAddPreEdgeTo(forExpr);
				//
				preEdges.push(CFEdge.Type.EPSILON);
				preNodes.push(forEnd);
			} else {
				// It's a traditional for-loop: 
				//   forInit? ';' expression? ';' forUpdate?
				CFNode forInit = null;
				if (ctx.forControl().forInit() != null) { // non-empty init
					forInit = new CFNode();
					forInit.setLineOfCode(ctx.forControl().forInit().getStart().getLine());
					forInit.setCode(getOriginalCodeText(ctx.forControl().forInit()));
					addContextualProperty(forInit, ctx.forControl().forInit());
					addNodeAndPreEdge(forInit);
				}
				// for-expression
				CFNode forExpr = new CFNode();
				if (ctx.forControl().expression() == null) {
					forExpr.setLineOfCode(ctx.forControl().getStart().getLine());
					forExpr.setCode("for ( ; )");
				} else {
					forExpr.setLineOfCode(ctx.forControl().expression().getStart().getLine());
					forExpr.setCode("for (" + getOriginalCodeText(ctx.forControl().expression()) + ")");
				}
				addContextualProperty(forExpr, ctx.forControl().expression());
				cfg.addVertex(forExpr);
				if (forInit != null)
					cfg.addEdge(new Edge<>(forInit, new CFEdge(CFEdge.Type.EPSILON), forExpr));
				else
					popAddPreEdgeTo(forExpr);
				// for-update
				CFNode forUpdate = new CFNode();
				if (ctx.forControl().forUpdate() == null) { // empty for-update
					forUpdate.setCode(" ; ");
					forUpdate.setLineOfCode(ctx.forControl().getStart().getLine());
				} else {
					forUpdate.setCode(getOriginalCodeText(ctx.forControl().forUpdate()));
					forUpdate.setLineOfCode(ctx.forControl().forUpdate().getStart().getLine());
				}
				addContextualProperty(forUpdate, ctx.forControl().forUpdate());
				cfg.addVertex(forUpdate);
				//
				CFNode forEnd = new CFNode();
				forEnd.setLineOfCode(-100*nextLine+25);
//				forEnd.setLineOfCode(get_end_block_line());
				forEnd.setCode("[endfor]");
				cfg.addVertex(forEnd);
				cfg.addEdge(new Edge<>(forExpr, new CFEdge(CFEdge.Type.FALSE), forEnd));
				//
				preEdges.push(CFEdge.Type.TRUE);
				preNodes.push(forExpr);
				loopBlocks.push(new Block(forUpdate, forEnd)); // NOTE: start is 'forUpdate'
				visit(ctx.statement());
				loopBlocks.pop();
				// add last statement to end for
				CFNode lastNode = preNodes.pop();
				cfg.addEdge(new Edge<>(lastNode, new CFEdge(CFEdge.Type.ENDLOOP), forEnd));
				preNodes.push(lastNode);
				popAddPreEdgeTo(forUpdate);

				cfg.addEdge(new Edge<>(forUpdate, new CFEdge(CFEdge.Type.EPSILON), forExpr));
//				cfg.addEdge(new Edge<>(forUpdate, new CFEdge(CFEdge.Type.EPSILON), forExpr));
				
				//
				preEdges.push(CFEdge.Type.EPSILON);
				preNodes.push(forEnd);
			}
			return null;
		}

		@Override
		public Void visitWhileStatement(JavaParser.WhileStatementContext ctx) {
			// add start while
			int nextLine = ctx.getStart().getLine();
			label_start_block( nextLine ,"[startwhile]");
			
			// add start while
//			CFNode startWhile = new CFNode();
//			startWhile.setLineOfCode(-5);
//			startWhile.setCode("[startwhile]");
//			addNodeAndPreEdge(startWhile);
//			preEdges.push(CFEdge.Type.EPSILON);
//			preNodes.push(startWhile);
			
			// 'while' parExpression statement
			CFNode whileNode = new CFNode();
			whileNode.setLineOfCode(ctx.getStart().getLine());
			whileNode.setCode("while " + getOriginalCodeText(ctx.parExpression()));
			addContextualProperty(whileNode, ctx);
			addNodeAndPreEdge(whileNode);
			//
			CFNode endwhile = new CFNode();
			endwhile.setLineOfCode(-100*nextLine+25);
//			endwhile.setLineOfCode(get_end_block_line());
			endwhile.setCode("[endwhile]");
			cfg.addVertex(endwhile);
			cfg.addEdge(new Edge<>(whileNode, new CFEdge(CFEdge.Type.FALSE), endwhile));
			//
			preEdges.push(CFEdge.Type.TRUE);
			preNodes.push(whileNode);
			loopBlocks.push(new Block(whileNode, endwhile));
			visit(ctx.statement());
			label_end_block(endwhile);
			loopBlocks.pop();
//			popAddPreEdgeTo(whileNode);
			//
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(endwhile);
			return null;
		}

		@Override
		public Void visitDoWhileStatement(JavaParser.DoWhileStatementContext ctx) {
			// add start do while
//			CFNode startWhile = new CFNode();
//			startWhile.setLineOfCode(-7);
//			startWhile.setCode("[start-do-while]");
//			addNodeAndPreEdge(startWhile);
//			preEdges.push(CFEdge.Type.EPSILON);
//			preNodes.push(startWhile);
			// add start while
			int nextLine = ctx.getStart().getLine();
			label_start_block( nextLine,"[start-do-while]" );
			
			// 'do' statement 'while' parExpression ';'
			CFNode doNode = new CFNode();
			doNode.setLineOfCode(ctx.getStart().getLine());
			doNode.setCode("do");
			addNodeAndPreEdge(doNode);
			//
			CFNode whileNode = new CFNode();
			whileNode.setLineOfCode(ctx.parExpression().getStart().getLine());
			whileNode.setCode("while " + getOriginalCodeText(ctx.parExpression()));
			addContextualProperty(whileNode, ctx);
			cfg.addVertex(whileNode);
			//
			CFNode doWhileEnd = new CFNode();
			doWhileEnd.setLineOfCode(-100*nextLine+25);
//			doWhileEnd.setLineOfCode(get_end_block_line());
			doWhileEnd.setCode("end-do-while");
			cfg.addVertex(doWhileEnd);
			//
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(doNode);
			loopBlocks.push(new Block(whileNode, doWhileEnd));
			visit(ctx.statement());
			label_end_block(doWhileEnd);
			loopBlocks.pop();
//			popAddPreEdgeTo(whileNode);
			cfg.addEdge(new Edge<>(whileNode, new CFEdge(CFEdge.Type.TRUE), doNode));
			cfg.addEdge(new Edge<>(whileNode, new CFEdge(CFEdge.Type.FALSE), doWhileEnd));
			//
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(doWhileEnd);
			return null;
		}

		@Override
		public Void visitSwitchStatement(JavaParser.SwitchStatementContext ctx) {
			
//			// add start Switch
//			CFNode startSwitch = new CFNode();
//			startSwitch.setLineOfCode(-9);
//			startSwitch.setCode("[start-switch]");
//			addNodeAndPreEdge(startSwitch);
//			preEdges.push(CFEdge.Type.EPSILON);
//			preNodes.push(startSwitch);
			
			// add start while
			int nextLine = ctx.getStart().getLine();
			label_start_block( nextLine ,"[start-switch]");
			
			// 'switch' parExpression '{' switchBlockStatementGroup* switchLabel* '}'
			CFNode switchNode = new CFNode();
			switchNode.setLineOfCode(ctx.getStart().getLine());
			switchNode.setCode("switch " + getOriginalCodeText(ctx.parExpression()));
			addContextualProperty(switchNode, ctx);
			addNodeAndPreEdge(switchNode);
			//
			CFNode endSwitch = new CFNode();
			endSwitch.setLineOfCode(-100*nextLine+25);
//			endSwitch.setLineOfCode(get_end_block_line());
			endSwitch.setCode("end-switch");
			cfg.addVertex(endSwitch);
			//
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(switchNode);
			loopBlocks.push(new Block(switchNode, endSwitch));
			//
			CFNode preCase = null;
			for (JavaParser.SwitchBlockStatementGroupContext grp: ctx.switchBlockStatementGroup()) {
				// switchBlockStatementGroup :  switchLabel+ blockStatement+
				preCase = visitSwitchLabels(grp.switchLabel(), preCase);
				for (JavaParser.BlockStatementContext blk: grp.blockStatement())
					visit(blk);
			}
			preCase = visitSwitchLabels(ctx.switchLabel(), preCase);
			
			label_end_block(endSwitch);
			loopBlocks.pop();
//			popAddPreEdgeTo(endSwitch);
			if (preCase != null)
				cfg.addEdge(new Edge<>(preCase, new CFEdge(CFEdge.Type.FALSE), endSwitch));
			//
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(endSwitch);
			return null;
		}

		private CFNode visitSwitchLabels(List<JavaParser.SwitchLabelContext> list, CFNode preCase) {
			//  switchLabel :  'case' constantExpression ':'  |  'case' enumConstantName ':'  |  'default' ':'
			CFNode caseStmnt = preCase;
			for (JavaParser.SwitchLabelContext ctx: list) {
				caseStmnt = new CFNode();
				caseStmnt.setLineOfCode(ctx.getStart().getLine());
				caseStmnt.setCode(getOriginalCodeText(ctx));
				caseStmnt.setProperty("is_simple_stmt",  true);
				cfg.addVertex(caseStmnt);
				if (dontPop)
					dontPop = false;
				else
					cfg.addEdge(new Edge<>(preNodes.pop(), new CFEdge(preEdges.pop()), caseStmnt));
//					preNodes.push(caseStmnt);
				if (preCase != null)
					cfg.addEdge(new Edge<>(preCase, new CFEdge(CFEdge.Type.FALSE), caseStmnt));
				if (ctx.getStart().getText().equals("default")) {
					preEdges.push(CFEdge.Type.EPSILON);
					preNodes.push(caseStmnt);
					caseStmnt = null;
				} else { // any other case ...
					dontPop = true;
					casesQueue.add(caseStmnt);
					preCase = caseStmnt;
//					preNodes.push(caseStmnt);
				}
			}
			return caseStmnt;
		}

		@Override
		public Void visitLabelStatement(JavaParser.LabelStatementContext ctx) {
			// add start while
			int nextLine = ctx.getStart().getLine();
			label_start_block( nextLine ,"[start-label]");
			
			// Identifier ':' statement
			// For each visited label-block, a Block object is created with 
			// the the current node as the start, and a dummy node as the end.
			// The newly created label-block is stored in an ArrayList of Blocks.
			CFNode labelNode = new CFNode();
			labelNode.setLineOfCode(ctx.getStart().getLine());
			labelNode.setCode(ctx.Identifier() + ": ");
			addContextualProperty(labelNode, ctx);
			addNodeAndPreEdge(labelNode);
			//
			CFNode endLabelNode = new CFNode();
			endLabelNode.setLineOfCode(-100*nextLine+25);
//			endLabelNode.setLineOfCode(0);
			endLabelNode.setCode("end-label");
			cfg.addVertex(endLabelNode);
			//
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(labelNode);
			labeledBlocks.add(new Block(labelNode, endLabelNode, ctx.Identifier().getText()));
			visit(ctx.statement());
			label_end_block(endLabelNode);
//			popAddPreEdgeTo(endLabelNode);
			//
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(endLabelNode);
			return null;
		}

		@Override
		public Void visitReturnStatement(JavaParser.ReturnStatementContext ctx) {
			// 'return' expression? ';'
			CFNode ret = new CFNode();
			ret.setLineOfCode(ctx.getStart().getLine());
			ret.setCode(getOriginalCodeText(ctx));

			ret.setProperty("is_simple_stmt",  true);
			addContextualProperty(ret, ctx);
			addNodeAndPreEdge(ret);
			dontPop = true;
			return null;
		}

		@Override
		public Void visitBreakStatement(JavaParser.BreakStatementContext ctx) {
			// 'break' Identifier? ';'
			// if a label is specified, search for the corresponding block in the labels-list,
			// and create an epsilon edge to the end of the labeled-block; else
			// create an epsilon edge to the end of the loop-block on top of the loopBlocks stack.
			CFNode breakNode = new CFNode();
			breakNode.setLineOfCode(ctx.getStart().getLine());
			breakNode.setCode(getOriginalCodeText(ctx));
			breakNode.setProperty("is_simple_stmt",  true);
			
			addContextualProperty(breakNode, ctx);
			addNodeAndPreEdge(breakNode);
			if (ctx.Identifier() != null) {
				// a label is specified
				for (Block block: labeledBlocks) {
					if (block.label.equals(ctx.Identifier().getText())) {
						cfg.addEdge(new Edge<>(breakNode, new CFEdge(CFEdge.Type.EPSILON), block.end));
						break;
					}
				}
			} else {
				// no label
				Block block = loopBlocks.peek();
				cfg.addEdge(new Edge<>(breakNode, new CFEdge(CFEdge.Type.EPSILON), block.end));
			}
			dontPop = true;
			return null;
		}

		@Override
		public Void visitContinueStatement(JavaParser.ContinueStatementContext ctx) {
			// 'continue' Identifier? ';'
			// if a label is specified, search for the corresponding block in the labels-list,
			// and create an epsilon edge to the start of the labeled-block; else
			// create an epsilon edge to the start of the loop-block on top of the loopBlocks stack.
			CFNode continueNode = new CFNode();
			continueNode.setLineOfCode(ctx.getStart().getLine());
			continueNode.setCode(getOriginalCodeText(ctx));
			continueNode.setProperty("is_simple_stmt",  true);
			
			addContextualProperty(continueNode, ctx);
			addNodeAndPreEdge(continueNode);
			if (ctx.Identifier() != null) {  
				// a label is specified
				for (Block block: labeledBlocks) {
					if (block.label.equals(ctx.Identifier().getText())) {
						cfg.addEdge(new Edge<>(continueNode, new CFEdge(CFEdge.Type.EPSILON), block.start));
						break;
					}
				}
			} else {  
				// no label
				Block block = loopBlocks.peek();
				cfg.addEdge(new Edge<>(continueNode, new CFEdge(CFEdge.Type.EPSILON), block.start));
			}
			dontPop = true;
			return null;
		}

		@Override
		public Void visitSynchBlockStatement(JavaParser.SynchBlockStatementContext ctx) {
			// add SynchBlockStatement
			int nextLine = ctx.getStart().getLine();
			label_start_block( nextLine ,"[start-synchronized]");
			// 'synchronized' parExpression block
			CFNode syncStmt = new CFNode();
			syncStmt.setLineOfCode(ctx.getStart().getLine());
			syncStmt.setCode("synchronized " + getOriginalCodeText(ctx.parExpression()));
			addContextualProperty(syncStmt, ctx);
			addNodeAndPreEdge(syncStmt);
			//
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(syncStmt);
			visit(ctx.block());
			//
			CFNode endSyncBlock = new CFNode();
			endSyncBlock.setLineOfCode(0);
			endSyncBlock.setLineOfCode(-100*nextLine+25);
//			endSyncBlock.setCode("[end-synchronized]");
			addNodeAndPreEdge(endSyncBlock);
			//
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(endSyncBlock);
			return null;
		}

		@Override
		public Void visitTryStatement(JavaParser.TryStatementContext ctx) {
			int nextLine = ctx.getStart().getLine();
			label_start_block( nextLine ,"[start-exception]");
			
			// 'try' block (catchClause+ finallyBlock? | finallyBlock)
			CFNode tryNode = new CFNode();
			tryNode.setLineOfCode(ctx.getStart().getLine());
			tryNode.setCode("try");
			addContextualProperty(tryNode, ctx);
			addNodeAndPreEdge(tryNode);
			//
			CFNode endTry = new CFNode();
			endTry.setLineOfCode(-1);
			endTry.setCode("end-try");
			cfg.addVertex(endTry);
			//
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(tryNode);
			tryBlocks.push(new Block(tryNode, endTry));
			visit(ctx.block());
			popAddPreEdgeTo(endTry);

			// If there is a finally-block, visit it first
			CFNode finallyNode = null;
			CFNode endFinally = null;
			if (ctx.finallyBlock() != null) {
				// 'finally' block
				finallyNode = new CFNode();
				finallyNode.setLineOfCode(ctx.finallyBlock().getStart().getLine());
				finallyNode.setCode("finally");
				addContextualProperty(finallyNode, ctx.finallyBlock());
				cfg.addVertex(finallyNode);
				cfg.addEdge(new Edge<>(endTry, new CFEdge(CFEdge.Type.EPSILON), finallyNode));
				//
				preEdges.push(CFEdge.Type.EPSILON);
				preNodes.push(finallyNode);
				visit(ctx.finallyBlock().block());
				//
				endFinally = new CFNode();
				endFinally.setLineOfCode(-2);
				endFinally.setCode("end-finally");
				addNodeAndPreEdge(endFinally);
			}
			// Now visit any available catch clauses
			if (ctx.catchClause() != null && ctx.catchClause().size() > 0) {
				// 'catch' '(' variableModifier* catchType Identifier ')' block
				CFNode catchNode;
				CFNode endCatch = new CFNode();
				endCatch.setLineOfCode(-3);
				endCatch.setCode("end-catch");
				cfg.addVertex(endCatch);
				for (JavaParser.CatchClauseContext cx: ctx.catchClause()) {
					// connect the try-node to all catch-nodes;
					// create a single end-catch for all catch-blocks;
					catchNode = new CFNode();
					catchNode.setLineOfCode(cx.getStart().getLine());
					catchNode.setCode("catch (" + cx.catchType().getText() + " " + cx.Identifier().getText() + ")");
					addContextualProperty(catchNode, cx);
					cfg.addVertex(catchNode);
					cfg.addEdge(new Edge<>(endTry, new CFEdge(CFEdge.Type.THROWS), catchNode));
					//
					preEdges.push(CFEdge.Type.EPSILON);
					preNodes.push(catchNode);
					visit(cx.block());
					popAddPreEdgeTo(endCatch);
				}
				if (finallyNode != null) {
					// connect end-catch node to finally-node,
					// and push end-finally to the stack ...
					cfg.addEdge(new Edge<>(endCatch, new CFEdge(CFEdge.Type.EPSILON), finallyNode));
					preEdges.push(CFEdge.Type.EPSILON);
					preNodes.push(endFinally);
				} else {
					// connect end-catch node to end-try,
					// and push end-try to the the stack ...
					cfg.addEdge(new Edge<>(endCatch, new CFEdge(CFEdge.Type.EPSILON), endTry));
					preEdges.push(CFEdge.Type.EPSILON);
					preNodes.push(endTry);
				}
			} else {
				// No catch-clause; it's a try-finally
				// push end-finally to the stack ...
				preEdges.push(CFEdge.Type.EPSILON);
				preNodes.push(endFinally);
			}
			// NOTE that Java does not allow a singular try-block (without catch or finally)
			CFNode endExceptionNode = new CFNode();
			endExceptionNode.setLineOfCode(-100*nextLine+25);
//			endSwitch.setLineOfCode(get_end_block_line());
			endExceptionNode.setCode("[end-exception]");
//			label_end_block(endExceptionNode);
//			popAddPreEdgeTo(endExceptionNode);
			addNodeAndPreEdge(endExceptionNode);
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(endExceptionNode);
			
			return null;
		}
		
		@Override
		public Void visitTryWithResourceStatement(JavaParser.TryWithResourceStatementContext ctx) {
			// add start while
			int nextLine = ctx.getStart().getLine();
			label_start_block( nextLine ,"[start-exception]");
			// 'try' resourceSpecification block catchClause* finallyBlock?
			// resourceSpecification :  '(' resources ';'? ')'
			// resources :  resource (';' resource)*
			// resource  :  variableModifier* classOrInterfaceType variableDeclaratorId '=' expression
			CFNode tryNode = new CFNode();
			tryNode.setLineOfCode(ctx.getStart().getLine());
			tryNode.setCode("try");
			addContextualProperty(tryNode, ctx);
			addNodeAndPreEdge(tryNode);
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(tryNode);
			//
			// Iterate over all resources ...
			for (JavaParser.ResourceContext rsrc: ctx.resourceSpecification().resources().resource()) {
				CFNode resource = new CFNode();
				resource.setLineOfCode(rsrc.getStart().getLine());
				resource.setCode(getOriginalCodeText(rsrc));
				//
				addContextualProperty(resource, rsrc);
				addNodeAndPreEdge(resource);
				//
				preEdges.push(CFEdge.Type.EPSILON);
				preNodes.push(resource);
			}
			//
			CFNode endTry = new CFNode();
			
			endTry.setLineOfCode(-1);
			endTry.setCode("end-try");
			cfg.addVertex(endTry);
			//
			tryBlocks.push(new Block(tryNode, endTry));
			visit(ctx.block());
			popAddPreEdgeTo(endTry);

			// If there is a finally-block, visit it first
			CFNode finallyNode = null;
			CFNode endFinally = null;
			if (ctx.finallyBlock() != null) {
				// 'finally' block
				finallyNode = new CFNode();
				finallyNode.setLineOfCode(ctx.finallyBlock().getStart().getLine());
				finallyNode.setCode("finally");
				addContextualProperty(finallyNode, ctx.finallyBlock());
				cfg.addVertex(finallyNode);
				cfg.addEdge(new Edge<>(endTry, new CFEdge(CFEdge.Type.EPSILON), finallyNode));
				//
				preEdges.push(CFEdge.Type.EPSILON);
				preNodes.push(finallyNode);
				visit(ctx.finallyBlock().block());
				//
				endFinally = new CFNode();
				endFinally.setLineOfCode(-2);
				endFinally.setCode("end-finally");
				addNodeAndPreEdge(endFinally);
			}
			// Now visit any available catch clauses
			if (ctx.catchClause() != null && ctx.catchClause().size() > 0) {
				// 'catch' '(' variableModifier* catchType Identifier ')' block
				CFNode catchNode;
				CFNode endCatch = new CFNode();
				endCatch.setLineOfCode(-3);
				endCatch.setCode("end-catch");
				cfg.addVertex(endCatch);
				for (JavaParser.CatchClauseContext cx: ctx.catchClause()) {
					// connect the try-node to all catch-nodes;
					// create a single end-catch for all catch-blocks;
					catchNode = new CFNode();
					catchNode.setLineOfCode(cx.getStart().getLine());
					catchNode.setCode("catch (" + cx.catchType().getText() + " " + cx.Identifier().getText() + ")");
					addContextualProperty(catchNode, cx);
					cfg.addVertex(catchNode);
					cfg.addEdge(new Edge<>(endTry, new CFEdge(CFEdge.Type.THROWS), catchNode));
					//
					preEdges.push(CFEdge.Type.EPSILON);
					preNodes.push(catchNode);
					visit(cx.block());
					popAddPreEdgeTo(endCatch);
				}
				if (finallyNode != null) {
					// connect end-catch node to finally-node,
					// and push end-finally to the stack ...
					cfg.addEdge(new Edge<>(endCatch, new CFEdge(CFEdge.Type.EPSILON), finallyNode));
					preEdges.push(CFEdge.Type.EPSILON);
					preNodes.push(endFinally);
				} else {
					// connect end-catch node to end-try,
					// and push end-try to the the stack ...
					cfg.addEdge(new Edge<>(endCatch, new CFEdge(CFEdge.Type.EPSILON), endTry));
					preEdges.push(CFEdge.Type.EPSILON);
					preNodes.push(endTry);
				}
			} else if (finallyNode != null) {
				// No catch-clause; it's a try-finally
				// push end-finally to the stack ...
				preEdges.push(CFEdge.Type.EPSILON);
				preNodes.push(endFinally);
			} else {
				// No catch-clause and no finally;
				// push end-try to the stack ...
				preEdges.push(CFEdge.Type.EPSILON);
				preNodes.push(endTry);
			}
			CFNode endExceptionNode = new CFNode();
			endExceptionNode.setLineOfCode(-100*nextLine+25);
//			endSwitch.setLineOfCode(get_end_block_line());
			endExceptionNode.setCode("[end-exception]");
//			label_end_block(endExceptionNode);
//			popAddPreEdgeTo(endExceptionNode);
			addNodeAndPreEdge(endExceptionNode);
			preEdges.push(CFEdge.Type.EPSILON);
			preNodes.push(endExceptionNode);
			return null;
		}

		@Override
		public Void visitThrowStatement(JavaParser.ThrowStatementContext ctx) {
			// 'throw' expression ';'
			CFNode throwNode = new CFNode();
			throwNode.setLineOfCode(ctx.getStart().getLine());
			throwNode.setCode("throw " + getOriginalCodeText(ctx.expression()));
			throwNode.setProperty("is_simple_stmt",  true);
			
			addContextualProperty(throwNode, ctx);
			addNodeAndPreEdge(throwNode);
			preNodes.push(throwNode);
			preEdges.push(CFEdge.Type.EPSILON);
//			dontPop = false;
			//
			if (!tryBlocks.isEmpty()) {
				Block tryBlock = tryBlocks.peek();
				cfg.addEdge(new Edge<>(throwNode, new CFEdge(CFEdge.Type.THROWS), tryBlock.end));
			} else {
				// do something when it's a throw not in a try-catch block ...
				// in such a situation, the method declaration has a throws clause;
				// so we should create a special node for the method-throws, 
				// and create an edge from this throw-statement to that throws-node.
			}
			dontPop = true;
			return null;
		}

		/**
		 * Get resulting Control-Flow-Graph of this CFG-Builder.
		 */
		public ControlFlowGraph getCFG() {
			return cfg;
		}

		/**
		 * Add this node to the CFG and create edge from pre-node to this node.
		 */
		private void addNodeAndPreEdge(CFNode node) {
			cfg.addVertex(node);
			popAddPreEdgeTo(node);
		}

		/**
		 * Add a new edge to the given node, by poping the edge-type of the stack.
		 */
		private void popAddPreEdgeTo(CFNode node) {
			if (dontPop)
				dontPop = false;
			else {
				Logger.debug("\nPRE-NODES = " + preNodes.size());
				Logger.debug("PRE-EDGES = " + preEdges.size() + '\n');
				cfg.addEdge(new Edge<>(preNodes.pop(), new CFEdge(preEdges.pop()), node));
			}
			//
			for (int i = casesQueue.size(); i > 0; --i)
				cfg.addEdge(new Edge<>(casesQueue.remove(), new CFEdge(CFEdge.Type.TRUE), node));
		}

		/**
		 * Get the original program text for the given parser-rule context.
		 * This is required for preserving whitespaces.
		 */
		private String getOriginalCodeText(ParserRuleContext ctx) {
			int start = ctx.start.getStartIndex();
			int stop = ctx.stop.getStopIndex();
			Interval interval = new Interval(start, stop);
			return ctx.start.getInputStream().getText(interval);
		}

		/**
		 * A simple structure for holding the start, end, and label of code blocks.
		 * These are used to handle 'break' and 'continue' statements.
		 */
		private class Block {

			public final String label;
			public final CFNode start, end;

			Block(CFNode start, CFNode end, String label) {
				this.start = start;
				this.end = end;
				this.label = label;
			}

			Block(CFNode start, CFNode end) {
				this(start, end, "");
			}
		}
	}
}
