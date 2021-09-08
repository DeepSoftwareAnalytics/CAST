/*** In The Name of Allah ***/
package ghaffarian.progex.java;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.Map;

import org.antlr.v4.gui.TreeViewer;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.misc.Interval;
import org.antlr.v4.runtime.tree.ParseTree;

import ghaffarian.progex.graphs.ast.ASEdge;
import ghaffarian.progex.graphs.ast.ASNode;
import ghaffarian.progex.graphs.ast.AbstractSyntaxTree;
import ghaffarian.progex.java.parser.JavaBaseVisitor;
import ghaffarian.progex.java.parser.JavaLexer;
import ghaffarian.progex.java.parser.JavaParser;
import ghaffarian.graphs.Edge;
import ghaffarian.nanologger.Logger;
import java.util.LinkedHashMap;
import java.util.List;

import org.antlr.v4.runtime.tree.TerminalNode;

/**
 * Abstract Syntax Tree (AST) builder for Java programs.
 * A Java parser generated via ANTLRv4 is used for this purpose.
 * This implementation is based on ANTLRv4's Visitor pattern.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class JavaASTBuilder {
	
	/**
	 * ‌Build and return the Abstract Syntax Tree (AST) for the given Java source file.
	 */
	public static AbstractSyntaxTree build(String javaFile, String nodeLevel) throws IOException {
		return build(new File(javaFile),nodeLevel);
	}
	
	/**
	 * ‌Build and return the Abstract Syntax Tree (AST) for the given Java source file.
	 */
	public static AbstractSyntaxTree build(File javaFile, String nodeLevel) throws IOException {
		if (!javaFile.getName().endsWith(".java"))
			throw new IOException("Not a Java File!");
		InputStream inFile = new FileInputStream(javaFile);
		ANTLRInputStream input = new ANTLRInputStream(inFile);
		JavaLexer lexer = new JavaLexer(input);
		CommonTokenStream tokens = new CommonTokenStream(lexer);
		JavaParser parser = new JavaParser(tokens);
		ParseTree tree = parser.compilationUnit();
        //show AST in console
//      System.out.println(tree.toStringTree(parser));

//      show AST in GUI
//      TreeViewer viewr = new TreeViewer(Arrays.asList(
//              parser.getRuleNames()),tree);
//      viewr.open();
		return build(javaFile.getPath(), tree, null, null,nodeLevel,tokens);
	}
	
	/**
	 * ‌Build and return the Abstract Syntax Tree (AST) for the given Parse-Tree.
	 * The 'ctxProps' map includes contextual-properties for particular nodes 
	 * in the parse-tree, which can be used for linking this graph with other 
	 * graphs by using the same parse-tree and the same contextual-properties.
	 */
	public static AbstractSyntaxTree build(String filePath, ParseTree tree, 
            String propKey, Map<ParserRuleContext, Object> ctxProps,  String nodeLevel,CommonTokenStream tokens) {
		AbstractSyntaxVisitor visitor = new AbstractSyntaxVisitor(filePath, propKey, ctxProps, nodeLevel);
        Logger.debug("Visitor building AST of: " + filePath);
        return visitor.build(tree,tokens);
	}
	
	/**
	 * Visitor class which constructs the AST for a given ParseTree.
	 */
	private static class AbstractSyntaxVisitor extends JavaBaseVisitor<String> {
        
		private CommonTokenStream tokenStream;
        private String propKey;
        private String typeModifier;
        private String memberModifier;
        private String nodeLevel;
        private Deque<ASNode> parentStack;
        private Deque<ASNode> slicedParentStack;
        private Deque<AbstractSyntaxTree> presentSubTreeStack;
        private final AbstractSyntaxTree AST;
        private Map<ASNode, AbstractSyntaxTree> slicedAST;
        private Map<String, String> vars, fields, methods;
        private AbstractSyntaxTree presentSubTree;
		private int varsCounter, fieldsCounter, methodsCounter, stmtBlockCounter;
		private Map<ParserRuleContext, Object> contexutalProperties;
		
		public AbstractSyntaxVisitor(String filePath, String propKey, Map<ParserRuleContext, Object> ctxProps, String nodeLevel) {
            parentStack = new ArrayDeque<>();
            slicedParentStack = new ArrayDeque<>();
            presentSubTreeStack = new ArrayDeque<>();
            AST = new AbstractSyntaxTree(filePath);
            slicedAST = new LinkedHashMap<>();
            presentSubTree = new AbstractSyntaxTree(filePath,false);
//            presentSubTree = new AbstractSyntaxTree(filePath);
			this.propKey = propKey;
			this.nodeLevel = nodeLevel;
			contexutalProperties = ctxProps;
            vars = new LinkedHashMap<>();
            fields = new LinkedHashMap<>();
            methods = new LinkedHashMap<>();
            varsCounter = 0; fieldsCounter = 0; methodsCounter = 0;
		}
        
        public AbstractSyntaxTree build(ParseTree tree,CommonTokenStream tokens)  {
        	tokenStream = tokens;
            JavaParser.CompilationUnitContext rootCntx = (JavaParser.CompilationUnitContext) tree;
            AST.root.setCode(new File(AST.filePath).getName());
            parentStack.push(AST.root);
            if (rootCntx.packageDeclaration() != null)
                visit(rootCntx.packageDeclaration());
            //
            if (rootCntx.importDeclaration() != null && rootCntx.importDeclaration().size() > 0) {
                ASNode imports = new ASNode(ASNode.Type.IMPORTS);
                imports.setLineOfCode(rootCntx.importDeclaration(0).getStart().getLine());
                Logger.debug("Adding imports");
                AST.addVertex(imports);
                AST.addEdge(AST.root, imports);
                parentStack.push(imports);
                for (JavaParser.ImportDeclarationContext importCtx : rootCntx.importDeclaration())
                    visit(importCtx);
                parentStack.pop();
            }
            //
            if (rootCntx.typeDeclaration() != null)
                for (JavaParser.TypeDeclarationContext typeDecCtx : rootCntx.typeDeclaration())
                    visit(typeDecCtx);
//            parentStack.pop();
            vars.clear();
            fields.clear();
            methods.clear();
//            AST.export("DOT",".");
            AbstractSyntaxTree complete_sliced_AST = AST;

            for (AbstractSyntaxTree value : slicedAST.values()) {
            	for (ASNode node : value.getAllVertices())
            	{
            		complete_sliced_AST.addVertex(node);
            	}
            	for (Edge<ASNode, ASEdge> edge : value.getAllEdges())
            	complete_sliced_AST.addEdge(edge.source, edge.target);
            }
            return complete_sliced_AST;
        }

        public void initNestedBlock() 
        {            	
        	if(presentSubTree.getAllEdges().size() >  0 ) 
        	{
	    		ASNode  curentParentNode = slicedParentStack.peek();
	        	if (curentParentNode.getType() == ASNode.Type.BLOCK || curentParentNode.getType() == ASNode.Type.THEN) 
	        	{
	        		presentSubTreeStack.push(presentSubTree);
	            	presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath,false);
	
	        	}
	        	else 
	        	{
		    		slicedAST.put(slicedParentStack.pop(),presentSubTree);
		    		presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath, false);
	        	}
        	}
        }
        
        public void initPresentSubTree() {
        	if (presentSubTreeStack.peek() == null) {
        		presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath,false);
        	}
        	else
        	{
        		presentSubTree = presentSubTreeStack.pop();
        	}
        }
        
        //=====================================================================//
        //                           DECLARATIONS                              //
        //=====================================================================//        
        
        @Override
        public String visitPackageDeclaration(JavaParser.PackageDeclarationContext ctx) {
            // packageDeclaration :  annotation* 'package' qualifiedName ';'
        	  if (nodeLevel == "statement")
              { 
	        	ASNode node = new ASNode(ASNode.Type.PACKAGE);
	            node.setCode(ctx.qualifiedName().getText());
	            node.setLineOfCode(ctx.getStart().getLine());
	            Logger.debug("Adding package");
	            AST.addVertex(node);
	            AST.addEdge(parentStack.peek(), node);
	            
              }
        	  return "";
        }

        @Override
        public String visitImportDeclaration(JavaParser.ImportDeclarationContext ctx) {
            // importDeclaration :  'import' 'static'? qualifiedName ('.' '*')? ';'
        	  if (nodeLevel == "statement")
              { 
	            String qualifiedName = ctx.qualifiedName().getText();
	            int last = ctx.getChildCount() - 1;
	            if (ctx.getChild(last - 1).getText().equals("*")
	                    && ctx.getChild(last - 2).getText().equals(".")) {
	                qualifiedName +=  " .* " ;
	            }
	            ASNode node = new ASNode(ASNode.Type.IMPORT);
	            node.setCode(qualifiedName);
	            node.setLineOfCode(ctx.getStart().getLine());
	            Logger.debug("Adding import " + qualifiedName);
	            AST.addVertex(node);
	            AST.addEdge(parentStack.peek(), node);
              }
            return "";
        }

        @Override
        public String visitTypeDeclaration(JavaParser.TypeDeclarationContext ctx) {
            // typeDeclaration
            //    :   classOrInterfaceModifier* classDeclaration
            //    |   classOrInterfaceModifier* enumDeclaration
            //    |   classOrInterfaceModifier* interfaceDeclaration
            //    |   classOrInterfaceModifier* annotationTypeDeclaration
            //    |   ';'
            typeModifier = "";
            for (JavaParser.ClassOrInterfaceModifierContext modifierCtx : ctx.classOrInterfaceModifier())
                typeModifier += modifierCtx.getText() + " ";
            typeModifier = typeModifier.trim();
            visitChildren(ctx);
            return "";
        }

        @Override
        public String visitClassDeclaration(JavaParser.ClassDeclarationContext ctx) {
            // classDeclaration 
            //   :  'class' Identifier typeParameters? 
            //      ('extends' typeType)? ('implements' typeList)? classBody
        	ASNode classNode = new ASNode(ASNode.Type.CLASS);
        	if (nodeLevel == "statement")
            {
//	        	ASNode classNode = new ASNode(ASNode.Type.CLASS);
	            classNode.setLineOfCode(ctx.getStart().getLine());
	            Logger.debug("Adding class node");
	            AST.addVertex(classNode);
	            AST.addEdge(parentStack.peek(), classNode);
	            //modifier :public,private,protected
	            ASNode modifierNode = new ASNode(ASNode.Type.MODIFIER);
	            modifierNode.setCode(typeModifier);
	            modifierNode.setLineOfCode(ctx.getStart().getLine());
	            Logger.debug("Adding class modifier");
	            AST.addVertex(modifierNode);
	            AST.addEdge(classNode, modifierNode);
	            //class name
	            ASNode nameNode = new ASNode(ASNode.Type.NAME);
	            String className = ctx.Identifier().getText();
	            if (ctx.typeParameters() != null)
	                className += ctx.typeParameters().getText();
	            nameNode.setCode(className);
	            nameNode.setLineOfCode(ctx.getStart().getLine());
	            Logger.debug("Adding class name: " + className);
	            AST.addVertex(nameNode);
	            AST.addEdge(classNode, nameNode);
	            //
	            if (ctx.typeType() != null) {
	                ASNode extendsNode = new ASNode(ASNode.Type.EXTENDS);
	                extendsNode.setCode(ctx.typeType().getText());
	                extendsNode.setLineOfCode(ctx.typeType().getStart().getLine());
	                Logger.debug("Adding extends " + ctx.typeType().getText());
	                AST.addVertex(extendsNode);
	                AST.addEdge(classNode, extendsNode);
	            }
	            //
	            if (ctx.typeList() != null) {
	                ASNode implementsNode = new ASNode(ASNode.Type.IMPLEMENTS);
	                implementsNode.setLineOfCode(ctx.typeList().getStart().getLine());
	                Logger.debug("Adding implements node ");
	                AST.addVertex(implementsNode);
	                AST.addEdge(classNode, implementsNode);
	                for (JavaParser.TypeTypeContext type : ctx.typeList().typeType()) {
	                    ASNode node = new ASNode(ASNode.Type.INTERFACE);
	                    node.setCode(type.getText());
	                    node.setLineOfCode(type.getStart().getLine());
	                    Logger.debug("Adding interface " + type.getText());
	                    AST.addVertex(node);
	                    AST.addEdge(implementsNode, node);
	                }
	            }
	            parentStack.push(classNode);
	            visit(ctx.classBody());
	            parentStack.pop();
            }
        	   else if(nodeLevel == "block")
               {
        		   visit(ctx.classBody());
               }
            return "";
        }

        @Override
        public String visitClassBodyDeclaration(JavaParser.ClassBodyDeclarationContext ctx) {
            // classBodyDeclaration
            //   :  ';'
            //   |  'static'? block
            //   |   modifier* memberDeclaration
            //
            // memberDeclaration
            //    :   methodDeclaration
            //    |   genericMethodDeclaration
            //    |   fieldDeclaration
            //    |   constructorDeclaration
            //    |   genericConstructorDeclaration
            //    |   interfaceDeclaration
            //    |   annotationTypeDeclaration
            //    |   classDeclaration
            //    |   enumDeclaration
            //
            if (ctx.block() != null) {
                ASNode staticBlock = new ASNode(ASNode.Type.STATIC_BLOCK);
                staticBlock.setLineOfCode(ctx.block().getStart().getLine());
                Logger.debug("Adding static block");
                AST.addVertex(staticBlock);
                AST.addEdge(parentStack.peek(), staticBlock);
                parentStack.push(staticBlock);
                visitChildren(ctx.block());
                parentStack.pop();
            } else if (ctx.memberDeclaration() != null) {
                // Modifier
                memberModifier = "";
                for (JavaParser.ModifierContext modCtx: ctx.modifier())
                    memberModifier += modCtx.getText() + " ";
                memberModifier = memberModifier.trim();
                // Field member
                if (ctx.memberDeclaration().fieldDeclaration() != null) {
                    ASNode fieldNode = new ASNode(ASNode.Type.FIELD);
                    fieldNode.setLineOfCode(ctx.memberDeclaration().fieldDeclaration().getStart().getLine());
                    Logger.debug("Adding field node");
                    AST.addVertex(fieldNode);
                    AST.addEdge(parentStack.peek(), fieldNode);
                    parentStack.push(fieldNode);
                    visit(ctx.memberDeclaration().fieldDeclaration());
                    parentStack.pop();
                } else if (ctx.memberDeclaration().constructorDeclaration() != null) {
                    // Constructor member
                    ASNode constructorNode = new ASNode(ASNode.Type.CONSTRUCTOR);
                    constructorNode.setLineOfCode(ctx.memberDeclaration().constructorDeclaration().getStart().getLine());
                    Logger.debug("Adding constructor node");
                    AST.addVertex(constructorNode);
                    AST.addEdge(parentStack.peek(), constructorNode);
                    parentStack.push(constructorNode);
                    visit(ctx.memberDeclaration().constructorDeclaration());
                    parentStack.pop();
                } else if (ctx.memberDeclaration().methodDeclaration() != null) {
                    // Method member
                	if (nodeLevel == "statement") {
                		ASNode methodNode = new ASNode(ASNode.Type.METHOD);
                        methodNode.setLineOfCode(ctx.memberDeclaration().methodDeclaration().getStart().getLine());
                        Logger.debug("Adding method node");
                        AST.addVertex(methodNode);
                        AST.addEdge(parentStack.peek(), methodNode);
                        parentStack.push(methodNode);
                	}
                	else if(nodeLevel == "block") {
                		ASNode methodNode = new ASNode(ASNode.Type.METHOD_SIGN);
                        methodNode.setLineOfCode(ctx.memberDeclaration().methodDeclaration().getStart().getLine());
                        Logger.debug("Adding method node");
                        AST.addVertex(methodNode);
                        AST.addEdge(parentStack.peek(), methodNode);
                        
                        parentStack.push(methodNode);
                        
                        ASNode slicedMethodNode  = new ASNode(ASNode.Type.METHOD_SIGN);
                        slicedMethodNode .setLineOfCode(ctx.memberDeclaration().methodDeclaration().getStart().getLine());
                        slicedMethodNode .setProperty("is_sliced_root",true);
                        presentSubTree.addVertex(slicedMethodNode );
                        slicedParentStack.push(slicedMethodNode );
//                        presentSubTree.addEdge(parentStack.peek(), methodNode);
                	}
                    
                    
                    visit(ctx.memberDeclaration().methodDeclaration());
                    parentStack.pop();
                } else if (ctx.memberDeclaration().classDeclaration() != null) {
                    // Inner-type member
                    visitChildren(ctx.memberDeclaration());
                }
            }
            return "";
        }
        
        @Override
        public String visitConstructorDeclaration(JavaParser.ConstructorDeclarationContext ctx) {
            // constructorDeclaration :  Identifier formalParameters ('throws' qualifiedNameList)? constructorBody
            // constructorBody :  block
            //
            ASNode modifierNode = new ASNode(ASNode.Type.MODIFIER);
            modifierNode.setLineOfCode(ctx.getStart().getLine());
            modifierNode.setCode(memberModifier);
            AST.addVertex(modifierNode);
            AST.addEdge(parentStack.peek(), modifierNode);
            //
            if (ctx.formalParameters().formalParameterList() != null) {
                ASNode paramsNode = new ASNode(ASNode.Type.PARAMS);
                paramsNode.setLineOfCode(ctx.formalParameters().getStart().getLine());
                AST.addVertex(paramsNode);
                AST.addEdge(parentStack.peek(), paramsNode);
                parentStack.push(paramsNode);
                for (JavaParser.FormalParameterContext paramctx: 
                        ctx.formalParameters().formalParameterList().formalParameter()) {
                    ASNode varNode = new ASNode(ASNode.Type.VARIABLE);
                    varNode.setLineOfCode(paramctx.getStart().getLine());
                    AST.addVertex(varNode);
                    AST.addEdge(parentStack.peek(), varNode);
                    //
                    ASNode type = new ASNode(ASNode.Type.TYPE);
                    type.setCode(paramctx.typeType().getText());
                    type.setLineOfCode(paramctx.typeType().getStart().getLine());
                    AST.addVertex(type);
                    AST.addEdge(varNode, type);
                    //
                    ++varsCounter;
                    ASNode name = new ASNode(ASNode.Type.NAME);
                    String normalized = "$VARL_" + varsCounter;
                    vars.put(paramctx.variableDeclaratorId().Identifier().getText(), normalized);
                    name.setCode(paramctx.variableDeclaratorId().getText());
                    name.setNormalizedCode(normalized);
                    name.setLineOfCode(paramctx.variableDeclaratorId().getStart().getLine());
                    AST.addVertex(name);
                    AST.addEdge(varNode, name);
                }
                if (ctx.formalParameters().formalParameterList().lastFormalParameter() != null) {
                    ASNode varNode = new ASNode(ASNode.Type.VARIABLE);
                    varNode.setLineOfCode(ctx.formalParameters().formalParameterList().lastFormalParameter().getStart().getLine());
                    AST.addVertex(varNode);
                    AST.addEdge(parentStack.peek(), varNode);
                    //
                    ASNode type = new ASNode(ASNode.Type.TYPE);
                    type.setCode(ctx.formalParameters().formalParameterList().lastFormalParameter().typeType().getText());
                    type.setLineOfCode(ctx.formalParameters().formalParameterList().lastFormalParameter().typeType().getStart().getLine());
                    AST.addVertex(type);
                    AST.addEdge(varNode, type);
                    //
                    ++varsCounter;
                    ASNode name = new ASNode(ASNode.Type.NAME);
                    String normalized = "$VARL_" + varsCounter;
                    vars.put(ctx.formalParameters().formalParameterList().lastFormalParameter().variableDeclaratorId().Identifier().getText(), normalized);
                    name.setCode(ctx.formalParameters().formalParameterList().lastFormalParameter().variableDeclaratorId().getText());
                    name.setNormalizedCode(normalized);
                    name.setLineOfCode(ctx.formalParameters().formalParameterList().lastFormalParameter().variableDeclaratorId().getStart().getLine());
                    AST.addVertex(name);
                    AST.addEdge(varNode, name);
                }
                parentStack.pop();
            }
            //
            ASNode bodyBlock = new ASNode(ASNode.Type.BLOCK);
            bodyBlock.setLineOfCode(ctx.constructorBody().block().getStart().getLine());
            AST.addVertex(bodyBlock);
            AST.addEdge(parentStack.peek(), bodyBlock);
            parentStack.push(bodyBlock);
            visitChildren(ctx.constructorBody().block());
            parentStack.pop();
            resetLocalVars();
            return "";
        }

        @Override
        public String visitFieldDeclaration(JavaParser.FieldDeclarationContext ctx) {
            // fieldDeclaration    :  typeType variableDeclarators ';'
            // variableDeclarators :  variableDeclarator (',' variableDeclarator)*
            // variableDeclarator  :  variableDeclaratorId ('=' variableInitializer)?
            //
            for (JavaParser.VariableDeclaratorContext varctx : ctx.variableDeclarators().variableDeclarator()) {
                ASNode modifierNode = new ASNode(ASNode.Type.MODIFIER);
                modifierNode.setCode(memberModifier);
                modifierNode.setLineOfCode(ctx.getStart().getLine());
                AST.addVertex(modifierNode);
                AST.addEdge(parentStack.peek(), modifierNode);
                //
                ASNode type = new ASNode(ASNode.Type.TYPE);
                type.setCode(ctx.typeType().getText());
                type.setLineOfCode(ctx.typeType().getStart().getLine());
                AST.addVertex(type);
                AST.addEdge(parentStack.peek(), type);
                //
                ++fieldsCounter;
                ASNode name = new ASNode(ASNode.Type.NAME);
                String normalized = "$VARF_" + fieldsCounter;
                fields.put(varctx.variableDeclaratorId().Identifier().getText(), normalized);
                name.setCode(varctx.variableDeclaratorId().getText());
                name.setNormalizedCode(normalized);
                name.setLineOfCode(varctx.variableDeclaratorId().getStart().getLine());
                AST.addVertex(name);
                AST.addEdge(parentStack.peek(), name);
                //
                if (varctx.variableInitializer() != null) {
                    ASNode initNode = new ASNode(ASNode.Type.INIT_VALUE);
                    initNode.setCode( " = " + getOriginalCodeText(varctx.variableInitializer()));
                    initNode.setNormalizedCode( " = " + visit(varctx.variableInitializer()));
                    initNode.setLineOfCode(varctx.variableInitializer().getStart().getLine());
                    AST.addVertex(initNode);
                    AST.addEdge(parentStack.peek(), initNode);
                }
            }
            return "";
        }

        @Override
        public String visitMethodDeclaration(JavaParser.MethodDeclarationContext ctx) {
            //  methodDeclaration
            //  :   (typeType|'void') Identifier formalParameters ('[' ']')* ('throws' qualifiedNameList)?  ( methodBody | ';')
            //
            //  formalParameters :  '(' formalParameterList? ')'
            //
            //  formalParameterList
            //    :   formalParameter (',' formalParameter)* (',' lastFormalParameter)?
            //    |   lastFormalParameter
            //
            //  formalParameter :  variableModifier* typeType variableDeclaratorId
            //
            //  lastFormalParameter :  variableModifier* typeType '...' variableDeclaratorId
            //
        	if (nodeLevel == "statement") {
        		ASNode modifierNode = new ASNode(ASNode.Type.MODIFIER);
                modifierNode.setCode(memberModifier);
                modifierNode.setLineOfCode(ctx.getStart().getLine());
                Logger.debug("Adding method modifier");
                AST.addVertex(modifierNode);
                AST.addEdge(parentStack.peek(), modifierNode);
                //
                ASNode retNode = new ASNode(ASNode.Type.RETURN);
                retNode.setCode(ctx.getChild(0).getText());
                retNode.setLineOfCode(ctx.getStart().getLine());
                Logger.debug("Adding method type");
                AST.addVertex(retNode);
                AST.addEdge(parentStack.peek(), retNode);
                //
                ++methodsCounter;
                ASNode nameNode = new ASNode(ASNode.Type.NAME);
                String methodName = ctx.Identifier().getText();
                String normalized = "$METHOD_" + methodsCounter;
                methods.put(methodName, normalized);
                nameNode.setCode(methodName);
                nameNode.setNormalizedCode(normalized);
                nameNode.setLineOfCode(ctx.getStart().getLine());
                Logger.debug("Adding method name");
                AST.addVertex(nameNode);
                AST.addEdge(parentStack.peek(), nameNode);
                if (ctx.formalParameters().formalParameterList() != null) {
                    ASNode paramsNode = new ASNode(ASNode.Type.PARAMS);
                    paramsNode.setLineOfCode(ctx.formalParameters().getStart().getLine());
                    Logger.debug("Adding method params node");
                    AST.addVertex(paramsNode);
                    AST.addEdge(parentStack.peek(), paramsNode);
                    parentStack.push(paramsNode);
                    for (JavaParser.FormalParameterContext paramctx: 
                            ctx.formalParameters().formalParameterList().formalParameter()) {
                        ASNode varNode = new ASNode(ASNode.Type.VARIABLE);
                        varNode.setLineOfCode(paramctx.getStart().getLine());
                        AST.addVertex(varNode);
                        AST.addEdge(parentStack.peek(), varNode);
                        //
                        ASNode type = new ASNode(ASNode.Type.TYPE);
                        type.setCode(paramctx.typeType().getText());
                        type.setLineOfCode(paramctx.typeType().getStart().getLine());
                        AST.addVertex(type);
                        AST.addEdge(varNode, type);
                        //
                        ++varsCounter;
                        ASNode name = new ASNode(ASNode.Type.NAME);
                        normalized = "$VARL_" + varsCounter;
                        vars.put(paramctx.variableDeclaratorId().Identifier().getText(), normalized);
                        name.setCode(paramctx.variableDeclaratorId().getText());
                        name.setNormalizedCode(normalized);
                        name.setLineOfCode(paramctx.variableDeclaratorId().getStart().getLine());
                        AST.addVertex(name);
                        AST.addEdge(varNode, name);
                    }
                    if (ctx.formalParameters().formalParameterList().lastFormalParameter() != null) {
                        ASNode varNode = new ASNode(ASNode.Type.VARIABLE);
                        varNode.setLineOfCode(ctx.formalParameters().formalParameterList().lastFormalParameter().getStart().getLine());
                        AST.addVertex(varNode);
                        AST.addEdge(parentStack.peek(), varNode);
                        //
                        ASNode type = new ASNode(ASNode.Type.TYPE);
                        type.setCode(ctx.formalParameters().formalParameterList().lastFormalParameter().typeType().getText());
                        type.setLineOfCode(ctx.formalParameters().formalParameterList().lastFormalParameter().typeType().getStart().getLine());
                        AST.addVertex(type);
                        AST.addEdge(varNode, type);
                        //
                        ++varsCounter;
                        ASNode name = new ASNode(ASNode.Type.NAME);
                        normalized = "$VARL_" + varsCounter;
                        vars.put(ctx.formalParameters().formalParameterList().lastFormalParameter().variableDeclaratorId().Identifier().getText(), normalized);
                        name.setCode(ctx.formalParameters().formalParameterList().lastFormalParameter().variableDeclaratorId().getText());
                        name.setNormalizedCode(normalized);
                        name.setLineOfCode(ctx.formalParameters().formalParameterList().lastFormalParameter().variableDeclaratorId().getStart().getLine());
                        AST.addVertex(name);
                        AST.addEdge(varNode, name);
                    }
                    parentStack.pop();
                }
        	}
        	else if (nodeLevel == "block") {

        		ASNode modifierNode = new ASNode(ASNode.Type.MODIFIER);
                modifierNode.setCode(memberModifier);
                modifierNode.setLineOfCode(ctx.getStart().getLine());
                Logger.debug("Adding method modifier");

                presentSubTree.addVertex(modifierNode);
                presentSubTree.addEdge(slicedParentStack.peek(), modifierNode);
                //
                ASNode retNode = new ASNode(ASNode.Type.RETURN);
                retNode.setCode(ctx.getChild(0).getText());
                retNode.setLineOfCode(ctx.getStart().getLine());
                Logger.debug("Adding method type");
                presentSubTree.addVertex(retNode);
                presentSubTree.addEdge(slicedParentStack.peek(), retNode);
                //
                ++methodsCounter;
                ASNode nameNode = new ASNode(ASNode.Type.NAME);
                String methodName = ctx.Identifier().getText();
                String normalized = "$METHOD_" + methodsCounter;
                methods.put(methodName, normalized);
                nameNode.setCode(methodName);
                nameNode.setNormalizedCode(normalized);
                nameNode.setLineOfCode(ctx.getStart().getLine());
                Logger.debug("Adding method name");
                presentSubTree.addVertex(nameNode);
                presentSubTree.addEdge(slicedParentStack.peek(), nameNode);
                if (ctx.formalParameters().formalParameterList() != null) {
                    ASNode paramsNode = new ASNode(ASNode.Type.PARAMS);
                    paramsNode.setLineOfCode(ctx.formalParameters().getStart().getLine());
                    Logger.debug("Adding method params node");
                    presentSubTree.addVertex(paramsNode);
                    presentSubTree.addEdge(slicedParentStack.peek(), paramsNode);
                    slicedParentStack.push(paramsNode);
                    for (JavaParser.FormalParameterContext paramctx: 
                            ctx.formalParameters().formalParameterList().formalParameter()) {
                        ASNode varNode = new ASNode(ASNode.Type.VARIABLE);
                        varNode.setLineOfCode(paramctx.getStart().getLine());
                        presentSubTree.addVertex(varNode);
                        presentSubTree.addEdge(slicedParentStack.peek(), varNode);
                        //
                        ASNode type = new ASNode(ASNode.Type.TYPE);
                        type.setCode(paramctx.typeType().getText());
                        type.setLineOfCode(paramctx.typeType().getStart().getLine());
                        presentSubTree.addVertex(type);
                        presentSubTree.addEdge(varNode, type);
                        //
                        ++varsCounter;
                        ASNode name = new ASNode(ASNode.Type.NAME);
                        normalized = "$VARL_" + varsCounter;
                        vars.put(paramctx.variableDeclaratorId().Identifier().getText(), normalized);
                        name.setCode(paramctx.variableDeclaratorId().getText());
                        name.setNormalizedCode(normalized);
                        name.setLineOfCode(paramctx.variableDeclaratorId().getStart().getLine());
                        presentSubTree.addVertex(name);
                        presentSubTree.addEdge(varNode, name);
                    }
                    if (ctx.formalParameters().formalParameterList().lastFormalParameter() != null) {
                        ASNode varNode = new ASNode(ASNode.Type.VARIABLE);
                        varNode.setLineOfCode(ctx.formalParameters().formalParameterList().lastFormalParameter().getStart().getLine());
                        presentSubTree.addVertex(varNode);
                        presentSubTree.addEdge(slicedParentStack.peek(), varNode);
                        //
                        ASNode type = new ASNode(ASNode.Type.TYPE);
                        type.setCode(ctx.formalParameters().formalParameterList().lastFormalParameter().typeType().getText());
                        type.setLineOfCode(ctx.formalParameters().formalParameterList().lastFormalParameter().typeType().getStart().getLine());
                        presentSubTree.addVertex(type);
                        presentSubTree.addEdge(varNode, type);
                        //
                        ++varsCounter;
                        ASNode name = new ASNode(ASNode.Type.NAME);
                        normalized = "$VARL_" + varsCounter;
                        vars.put(ctx.formalParameters().formalParameterList().lastFormalParameter().variableDeclaratorId().Identifier().getText(), normalized);
                        name.setCode(ctx.formalParameters().formalParameterList().lastFormalParameter().variableDeclaratorId().getText());
                        name.setNormalizedCode(normalized);
                        name.setLineOfCode(ctx.formalParameters().formalParameterList().lastFormalParameter().variableDeclaratorId().getStart().getLine());
                        presentSubTree.addVertex(name);
                        presentSubTree.addEdge(varNode, name);
                    }
                    slicedParentStack.pop();
                }
        	slicedAST.put(slicedParentStack.pop(),presentSubTree);
        	presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath, false);
        	}
            if (ctx.methodBody() != null) {
            	if (nodeLevel == "statement") {
	        		ASNode methodBody = new ASNode(ASNode.Type.BLOCK);
	                methodBody.setLineOfCode(ctx.methodBody().getStart().getLine());
	                Logger.debug("Adding method block");
	                AST.addVertex(methodBody);
	                AST.addEdge(parentStack.peek(), methodBody);
	                parentStack.push(methodBody);
	                visitChildren(ctx.methodBody());
	                parentStack.pop();
	                resetLocalVars();
            	}
            	else if(nodeLevel == "block") {
            		ASNode methodBody = new ASNode(ASNode.Type.METHOD_BODY);
                    methodBody.setLineOfCode(ctx.methodBody().getStart().getLine());
                    Logger.debug("Adding method block");
                    AST.addVertex(methodBody);
                    AST.addEdge(parentStack.peek(), methodBody);
                    parentStack.push(methodBody);
//                    slicedParentStack.push(methodBody);
                    //
                    
//            		ASNode slicedMethodBody = new ASNode(ASNode.Type.METHOD_BODY);
//            		slicedMethodBody.setLineOfCode(ctx.methodBody().getStart().getLine());
//            		slicedMethodBody.setProperty("is_sliced_root",true);
//                    presentSubTree.addVertex(slicedMethodBody);
//                    slicedParentStack.push(slicedMethodBody);
                    
                    visitChildren(ctx.methodBody());
                    if (presentSubTree.getAllVertices().size() > 0) {
                    	slicedAST.put(slicedParentStack.pop(),presentSubTree);
                    	presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath, false);
                    	
                    }
                    parentStack.pop();
//                    slicedParentStack.pop();
                    resetLocalVars();
            	}
            	
            }
            return "";
        }
        public void processStmtLocalVariableDeclaration(JavaParser.LocalVariableDeclarationContext ctx) {
        	for (JavaParser.VariableDeclaratorContext varctx: ctx.variableDeclarators().variableDeclarator()) {
                ASNode varNode = new ASNode(ASNode.Type.VARIABLE);
                varNode.setLineOfCode(varctx.getStart().getLine());
                AST.addVertex(varNode);
                AST.addEdge(parentStack.peek(), varNode);
                //
                ASNode typeNode = new ASNode(ASNode.Type.TYPE);
                typeNode.setCode(ctx.typeType().getText());
                typeNode.setLineOfCode(ctx.typeType().getStart().getLine());
                AST.addVertex(typeNode);
                AST.addEdge(varNode, typeNode);
                //
                ++varsCounter;
                ASNode nameNode = new ASNode(ASNode.Type.NAME);
                String normalized = "$VARL_" + varsCounter;
                vars.put(varctx.variableDeclaratorId().Identifier().getText(), normalized);
                nameNode.setCode(varctx.variableDeclaratorId().getText());
                nameNode.setNormalizedCode(normalized);
                nameNode.setLineOfCode(varctx.variableDeclaratorId().getStart().getLine());
                AST.addVertex(nameNode);
                AST.addEdge(varNode, nameNode);
                //
                if (varctx.variableInitializer() != null) {
                    ASNode initNode = new ASNode(ASNode.Type.INIT_VALUE);
                    initNode.setCode(" = " + getOriginalCodeText(varctx.variableInitializer()));
                    initNode.setNormalizedCode(" = " + visit(varctx.variableInitializer()));
                    initNode.setLineOfCode(varctx.variableInitializer().getStart().getLine());
                    AST.addVertex(initNode);
                    AST.addEdge(varNode, initNode);
                }
            }
        	
        }
        public void processLocalVariableDeclaration(JavaParser.LocalVariableDeclarationContext ctx, AbstractSyntaxTree tree )
        {
        	for (JavaParser.VariableDeclaratorContext varctx: ctx.variableDeclarators().variableDeclarator()) {
                ASNode varNode = new ASNode(ASNode.Type.VARIABLE);
                varNode.setLineOfCode(varctx.getStart().getLine());
                tree.addVertex(varNode);
               	try
            	{
               		
               		tree.addEdge(slicedParentStack.peek(), varNode);
            	}
            	catch (Exception e)
            	{
            		//            	//++stmtBlockCounter;
                    //String normalizedBlock = "$STMTBLOCK_" + stmtBlockCounter;
                	//slicedSTMTSNode.setNormalizedCode(normalizedBlock);
                	//STMTSNode.setNormalizedCode(normalizedBlock);
            		
                	++stmtBlockCounter;
                    String normalizedBlock = "$STMTBLOCK_" + stmtBlockCounter;
                    
            		ASNode slicedSTMTSNode = new ASNode(ASNode.Type.STMTS);
            		slicedSTMTSNode.setLineOfCode(ctx.getStart().getLine());
            		slicedSTMTSNode.setNormalizedCode(normalizedBlock);
            		presentSubTree.addVertex(slicedSTMTSNode);
            		slicedParentStack.push(slicedSTMTSNode);
            		
            		presentSubTree.addEdge(slicedParentStack.peek(), varNode);
            		
            		ASNode STMTSNode = new ASNode(ASNode.Type.STMTS);
            		STMTSNode.setLineOfCode(ctx.getStart().getLine());
            		STMTSNode.setNormalizedCode(normalizedBlock);
            		AST.addVertex(STMTSNode);
            		AST.addEdge(parentStack.peek(), STMTSNode);
            	}
//                tree.addEdge(slicedParentStack.peek(), varNode);
                //
                ASNode typeNode = new ASNode(ASNode.Type.TYPE);
                typeNode.setCode(ctx.typeType().getText());
                typeNode.setLineOfCode(ctx.typeType().getStart().getLine());
                tree.addVertex(typeNode);
                tree.addEdge(varNode, typeNode);
                //
                ++varsCounter;
                ASNode nameNode = new ASNode(ASNode.Type.NAME);
                String normalized = "$VARL_" + varsCounter;
                vars.put(varctx.variableDeclaratorId().Identifier().getText(), normalized);
                nameNode.setCode(varctx.variableDeclaratorId().getText());
                nameNode.setNormalizedCode(normalized);
                nameNode.setLineOfCode(varctx.variableDeclaratorId().getStart().getLine());
                tree.addVertex(nameNode);
                tree.addEdge(varNode, nameNode);
                //
                if (varctx.variableInitializer() != null) {
                    ASNode initNode = new ASNode(ASNode.Type.INIT_VALUE);
                    initNode.setCode(" = " + getOriginalCodeText(varctx.variableInitializer()));
                    initNode.setNormalizedCode(" = " + visit(varctx.variableInitializer()));
                    initNode.setLineOfCode(varctx.variableInitializer().getStart().getLine());
                    tree.addVertex(initNode);
                    tree.addEdge(varNode, initNode);
                }
            }
        }
        @Override
        public String visitLocalVariableDeclaration(JavaParser.LocalVariableDeclarationContext ctx) {
            // localVariableDeclaration :  variableModifier* typeType variableDeclarators
            // variableDeclarators      :  variableDeclarator (',' variableDeclarator)*
            // variableDeclarator       :  variableDeclaratorId ('=' variableInitializer)?
            //
        	  if (nodeLevel == "statement")
              {
        		  processStmtLocalVariableDeclaration(ctx);
              }            
              else if(nodeLevel == "block")
              {
            	  processLocalVariableDeclaration(ctx, presentSubTree);
              } 
            return "";
        }

        //=====================================================================//
        //                           STATEMENTS                                //
        //=====================================================================//
        
        private void visitStatement(ParserRuleContext ctx, String normalized) {
            Logger.printf(Logger.Level.DEBUG, "Visiting: (%d)  %s", ctx.getStart().getLine(), getOriginalCodeText(ctx));
            ASNode statementNode = new ASNode(ASNode.Type.STATEMENT);
            statementNode.setCode(getOriginalCodeText(ctx));
            statementNode.setNormalizedCode(normalized);
            statementNode.setLineOfCode(ctx.getStart().getLine());
            Logger.debug("Adding statement " + ctx.getStart().getLine());
            if (nodeLevel == "statement")
            {
            	AST.addVertex(statementNode);
                AST.addEdge(parentStack.peek(), statementNode);
            }
            else if(nodeLevel == "block") {
//            	presentSubTree.addVertex(slicedParentStack.peek());
            	try
            	{
            		presentSubTree.addVertex(statementNode);
            		presentSubTree.addEdge(slicedParentStack.peek(), statementNode);
            		
            	}
            	catch (Exception e)
            	{
            		presentSubTree.removeVertex(statementNode);
//            		presentSubTree.
            		++stmtBlockCounter;
            		String normalizedBlock = "$STMTBLOCK_" + stmtBlockCounter;
            	//++stmtBlockCounter;
                //String normalizedBlock = "$STMTBLOCK_" + stmtBlockCounter;
                //nameNode.setNormalizedCode(normalized);
            	//slicedSTMTSNode.setNormalizedCode(normalizedBlock);
            	//STMTSNode.setNormalizedCode(normalizedBlock);
            		ASNode slicedSTMTSNode = new ASNode(ASNode.Type.STMTS);
            		slicedSTMTSNode.setLineOfCode(ctx.getStart().getLine());
            		slicedSTMTSNode.setNormalizedCode(normalizedBlock);
            		presentSubTree.addVertex(slicedSTMTSNode);
            		slicedParentStack.push(slicedSTMTSNode);
            		presentSubTree.addVertex(statementNode);
            		presentSubTree.addEdge(slicedParentStack.peek(), statementNode);
            		
            		ASNode STMTSNode = new ASNode(ASNode.Type.STMTS);
            		STMTSNode.setLineOfCode(ctx.getStart().getLine());
            		STMTSNode.setNormalizedCode(normalizedBlock);
            		
//            		ASNode  curentParentNode = slicedParentStack.peek();
//            	 	if (curentParentNode != null && curentParentNode.getType() == ASNode.Type.BLOCK) {
//                		AbstractSyntaxTree subTree = presentSubTreeStack.pop();
//                		subTree.addVertex(STMTSNode);
//                		subTree.addEdge(slicedParentStack.peek(), STMTSNode);
//                		presentSubTreeStack.push(subTree);
//                	}
//                	else {
//                        this.AST.addVertex(STMTSNode);
//                        this.AST.addEdge(parentStack.peek(), STMTSNode);
//                	}
            		AST.addVertex(STMTSNode);
            		AST.addEdge(parentStack.peek(), STMTSNode);
            	}
          
            	
            }
        }
        
        @Override
        public String visitStatementExpression(JavaParser.StatementExpressionContext ctx) {
            // statementExpression :  expression
            visitStatement(ctx, visit(ctx.expression()));
            return "";
        }
        
        @Override
        public String visitBreakStatement(JavaParser.BreakStatementContext ctx) {
            if (ctx.Identifier() == null)
                visitStatement(ctx, null);
            else
                visitStatement(ctx, "break $LABEL");
            return "";
        }
        
        @Override
        public String visitContinueStatement(JavaParser.ContinueStatementContext ctx) {
            if (ctx.Identifier() == null)
                visitStatement(ctx, null);
            else
                visitStatement(ctx, "continue $LABEL");
            return "";
        }
        
        @Override
        public String visitReturnStatement(JavaParser.ReturnStatementContext ctx) {
            if (ctx.expression() == null)
                visitStatement(ctx, null);
            else
                visitStatement(ctx,  " return " + visit(ctx.expression()));
            return "";
        }
        
        @Override
        public String visitThrowStatement(JavaParser.ThrowStatementContext ctx) {
            visitStatement(ctx,  " throw " + visit(ctx.expression()));
            return "";
        }
        
        public void processSynchBlockStatement(JavaParser.SynchBlockStatementContext ctx,AbstractSyntaxTree tree)
        {
        	 ASNode slicedSynchNode = new ASNode(ASNode.Type.SYNC);
        	if ( nodeLevel=="block" ) {
//            	tree = new AbstractSyntaxTree(presentSubTree.filePath, false);
        		ASNode synchNode = new ASNode(ASNode.Type.SYNC);
                synchNode.setLineOfCode(ctx.getStart().getLine());
                synchNode.setProperty("is_sliced_root",true);
                
                ASNode  curentParentNode = slicedParentStack.peek();
                if (curentParentNode != null && curentParentNode.getType() == ASNode.Type.BLOCK) {
                	synchNode.setType(ASNode.Type.NESTED_SYNC );
                	slicedSynchNode.setType(ASNode.Type.NESTED_SYNC );
            		AbstractSyntaxTree subTree = presentSubTreeStack.pop();
            		subTree.addVertex(synchNode);
            		subTree.addEdge(slicedParentStack.peek(), synchNode);
            		presentSubTreeStack.push(subTree);
            	}
            	else {
                AST.addVertex(synchNode);
                AST.addEdge(parentStack.peek(), synchNode);
                slicedParentStack.push(synchNode);
            	}
        	}
           
            slicedSynchNode.setLineOfCode(ctx.getStart().getLine());
            tree.addVertex(slicedSynchNode);
            if(nodeLevel=="statement")
            	tree.addEdge(parentStack.peek(), slicedSynchNode);
            //
            slicedParentStack.push(slicedSynchNode);
            visitStatement(ctx.parExpression().expression(), visit(ctx.parExpression().expression()));
            slicedParentStack.pop();
            //
            ASNode block = new ASNode(ASNode.Type.BLOCK);
            block.setLineOfCode(ctx.block().getStart().getLine());
            tree.addVertex(block);
            tree.addEdge(slicedSynchNode, block);
            slicedParentStack.push(block);
            visit(ctx.block());
            slicedParentStack.pop();
        }
        @Override
        public String visitSynchBlockStatement(JavaParser.SynchBlockStatementContext ctx) {
            // synchBlockStatement :  'synchronized' parExpression block
            if (nodeLevel == "statement")
            {
            	processSynchBlockStatement(ctx, AST);
            	parentStack.pop();
            }            
            else if(nodeLevel == "block")
            {	
            		initNestedBlock();
            	    processSynchBlockStatement(ctx, presentSubTree);
                	if (presentSubTree.getAllEdges().size() >  0) 
                		slicedAST.put(slicedParentStack.pop(),presentSubTree);
                	initPresentSubTree();
//                    slicedAST.put(slicedParentStack.pop(),presentSubTree);
//                    initPresentSubTree();
//                    presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath,false);
            }
            return "";
        }
        
        public void processLabelStatement(JavaParser.LabelStatementContext ctx,AbstractSyntaxTree tree)
        {
        	ASNode slicedLabelNode = new ASNode(ASNode.Type.LABELED);
        	if ( nodeLevel=="block" ) {
//            	tree = new AbstractSyntaxTree(presentSubTree.filePath, false);
        		ASNode labelNode = new ASNode(ASNode.Type.LABELED);
                labelNode.setLineOfCode(ctx.getStart().getLine());
                labelNode.setProperty("is_sliced_root",true);
                
                ASNode  curentParentNode = slicedParentStack.peek();
                if (curentParentNode != null && curentParentNode.getType() == ASNode.Type.BLOCK) {
                	labelNode.setType(ASNode.Type.NESTED_LABELED);
                	slicedLabelNode.setType(ASNode.Type.NESTED_LABELED);
            		AbstractSyntaxTree subTree = presentSubTreeStack.pop();
            		subTree.addVertex(labelNode);
            		subTree.addEdge(slicedParentStack.peek(), labelNode);
            		presentSubTreeStack.push(subTree);
            	}
            	else {
	                AST.addVertex(labelNode);
	                AST.addEdge(parentStack.peek(), labelNode);
            	}
                slicedParentStack.push(labelNode);
        	}
        	 
             slicedLabelNode.setLineOfCode(ctx.getStart().getLine());
             tree.addVertex(slicedLabelNode);
             if(nodeLevel=="statement")
            	 tree.addEdge(parentStack.peek(), slicedLabelNode);
             //
             ASNode labelName = new ASNode(ASNode.Type.NAME);
             labelName.setCode(ctx.Identifier().getText());
             labelName.setNormalizedCode("$LABEL");
             labelName.setLineOfCode(ctx.getStart().getLine());
             tree.addVertex(labelName);
             tree.addEdge(slicedLabelNode, labelName);
             //
             

             
//             slicedParentStack.push(slicedLabelNode);
             slicedParentStack.push(slicedLabelNode);
             
             visit(ctx.statement());
//             slicedParentStack.pop();
//             ASNode slicedLabelBlockNode = new ASNode(ASNode.Type.BLOCK);
//             slicedLabelBlockNode.setLineOfCode(ctx.getStart().getLine());
//             tree.addVertex(slicedLabelBlockNode);
//             tree.addEdge(slicedLabelNode,slicedLabelBlockNode);
//             slicedParentStack.push(slicedLabelBlockNode);
        }
        @Override
        public String visitLabelStatement(JavaParser.LabelStatementContext ctx) {
            // labelStatement :  Identifier ':' statement
            if (nodeLevel == "statement")
            {
            	processLabelStatement(ctx, AST);
            	parentStack.pop();
            }            
            else if(nodeLevel == "block")
            {	
            		initNestedBlock();
            	    processLabelStatement(ctx, presentSubTree);
                	if (presentSubTree.getAllEdges().size() >  0) 
                		slicedAST.put(slicedParentStack.pop(),presentSubTree);
                	initPresentSubTree();
//                    slicedAST.put(slicedParentStack.pop(),presentSubTree);
//                    initPresentSubTree();
//                    presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath,false);
            } 
            return "";
        }
        
        public void processIfStatement(JavaParser.IfStatementContext ctx, AbstractSyntaxTree tree,Deque<ASNode> currentStack) 
        {
//        	slicedParentStack.pop();
        	ASNode slicedIfNode = new ASNode(ASNode.Type.IF);
        	slicedIfNode.setLineOfCode(ctx.getStart().getLine());
        	if ( nodeLevel=="block" ) {
//            	tree = new AbstractSyntaxTree(presentSubTree.filePath, false);
        		ASNode ifNode = new ASNode(ASNode.Type.IF);
                ifNode.setLineOfCode(ctx.getStart().getLine());
                ifNode.setProperty("is_sliced_root",true);
                
                // whether it is Nested_If_statement
                ASNode  curentParentNode = slicedParentStack.peek();
                if (curentParentNode != null && curentParentNode.getType() == ASNode.Type.BLOCK && presentSubTreeStack.peek() != null) {
                	ifNode.setType(ASNode.Type.NESTED_IF);
                	slicedIfNode.setType(ASNode.Type.NESTED_IF);
            		AbstractSyntaxTree subTree = presentSubTreeStack.pop();
            		subTree.addVertex( ifNode);
            		subTree.addEdge(slicedParentStack.peek(),  ifNode);
            		presentSubTreeStack.push(subTree);
            	}
            	else {
                AST.addVertex(ifNode);
                AST.addEdge(parentStack.peek(), ifNode);
            	}
                slicedParentStack.push(ifNode);
        	}
        	
        	
            tree.addVertex(slicedIfNode);
            if (nodeLevel == "statement")
            	tree.addEdge(parentStack.peek(), slicedIfNode);
            
            //
            ASNode cond = new ASNode(ASNode.Type.CONDITION);
            cond.setCode(getOriginalCodeText(ctx.parExpression().expression()));
            cond.setNormalizedCode(visit(ctx.parExpression().expression()));
            cond.setLineOfCode(ctx.parExpression().getStart().getLine());
            tree.addVertex(cond);
            tree.addEdge(slicedIfNode, cond);
            //
            ASNode thenNode = new ASNode(ASNode.Type.BLOCK);
//            ASNode thenNode = new ASNode(ASNode.Type.THEN);
            thenNode.setLineOfCode(ctx.statement(0).getStart().getLine());
            tree.addVertex(thenNode);
            tree.addEdge(slicedIfNode, thenNode);
            currentStack.push(thenNode);
            visit(ctx.statement(0));
            currentStack.pop();
            //
            if (ctx.statement(1) != null) {
                ASNode elseNode = new ASNode(ASNode.Type.ELSE);
                elseNode.setLineOfCode(ctx.statement(1).getStart().getLine());
                tree.addVertex(elseNode);
                tree.addEdge(slicedIfNode, elseNode);
//                currentStack.push(elseNode);
                
                ASNode elseBlock = new ASNode(ASNode.Type.BLOCK);
                elseBlock.setLineOfCode(ctx.statement(1).getStart().getLine());
                tree.addVertex(elseBlock);
                tree.addEdge(elseNode, elseBlock);
                slicedParentStack.push(elseBlock);
                visit(ctx.statement(1));
                slicedParentStack.pop();
                
//                visit(ctx.statement(1));
////                if  ( nodeLevel=="block" )
//                currentStack.pop();
            }
           
        
        }
        @Override
        public String visitIfStatement(JavaParser.IfStatementContext ctx) {
            // 'if' parExpression statement ('else' statement)?
            if (nodeLevel == "statement")
            {
            	processIfStatement(ctx, AST,parentStack);
            	parentStack.pop();
            }            
            else if(nodeLevel == "block")
            {	
            	
            	initNestedBlock();
            	
//            	slicedParentStack.pop();
            	// process IF block and add it to slicedAST
            	processIfStatement(ctx, presentSubTree,slicedParentStack);
            	if (presentSubTree.getAllEdges().size() >  0) 
            		slicedAST.put(slicedParentStack.pop(),presentSubTree);
            	initPresentSubTree();
            
//            	presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath,false);
            } 
            
            return "";

        }
        
        public void processingForstatement(JavaParser.ForStatementContext ctx, AbstractSyntaxTree tree )
        {
        	 
        	ASNode slicedForNode;
        	
        	if (ctx.forControl().enhancedForControl() != null) {
                // for-each loop
//        		ASNode forNode;
//                forNode = new ASNode(ASNode.Type.FOR_EACH);
//                forNode.setLineOfCode(ctx.getStart().getLine());
//                this.AST.addVertex(forNode);
//                this.AST.addEdge(parentStack.peek(), forNode);
        		slicedForNode = new ASNode(ASNode.Type.FOR_EACH);
            	if ( nodeLevel=="block" )
            	{
            		
            		ASNode forNode;
            		
                    forNode = new ASNode(ASNode.Type.FOR_EACH);
                    forNode.setLineOfCode(ctx.getStart().getLine());
                    forNode.setProperty("is_sliced_root",true);

                	ASNode  curentParentNode = slicedParentStack.peek();
                	if (curentParentNode != null && curentParentNode.getType() == ASNode.Type.BLOCK) {
                		forNode.setType(ASNode.Type.NESTED_FOR);
                		slicedForNode.setType(ASNode.Type.NESTED_FOR);
                		AbstractSyntaxTree subTree = presentSubTreeStack.pop();
                		subTree.addVertex(forNode);
                		subTree.addEdge(slicedParentStack.peek(), forNode);
                		presentSubTreeStack.push(subTree);
                	}
                	else {
	                    this.AST.addVertex(forNode);
	                    this.AST.addEdge(parentStack.peek(), forNode);
                	}
                    slicedParentStack.push(forNode);
                }
            	
                
             
                slicedForNode.setLineOfCode(ctx.getStart().getLine());
                slicedForNode.setProperty("is_sliced_root",true);
                tree.addVertex(slicedForNode);
                if(nodeLevel == "statement")
                	tree.addEdge(parentStack.peek(), slicedForNode);
               
                //
                ASNode varType = new ASNode(ASNode.Type.TYPE);
                varType.setCode(ctx.forControl().enhancedForControl().typeType().getText());
                varType.setLineOfCode(ctx.forControl().enhancedForControl().typeType().getStart().getLine());
                tree.addVertex(varType);
                tree.addEdge(slicedForNode, varType);
                //
                ++varsCounter;
                ASNode varID = new ASNode(ASNode.Type.NAME);
                String normalized = "$VARL_" + varsCounter;
                vars.put(ctx.forControl().enhancedForControl().variableDeclaratorId().Identifier().getText(), normalized);
                varID.setCode(ctx.forControl().enhancedForControl().variableDeclaratorId().getText());
                varID.setNormalizedCode(normalized);
                varID.setLineOfCode(ctx.forControl().enhancedForControl().variableDeclaratorId().getStart().getLine());
                tree.addVertex(varID);
                tree.addEdge(slicedForNode, varID);
                //
                ASNode expr = new ASNode(ASNode.Type.IN);
                expr.setCode(getOriginalCodeText(ctx.forControl().enhancedForControl().expression()));
                expr.setNormalizedCode(visit(ctx.forControl().enhancedForControl().expression()));
                expr.setLineOfCode(ctx.forControl().enhancedForControl().expression().getStart().getLine());
                tree.addVertex(expr);
                tree.addEdge(slicedForNode, expr);
            } 
            // Classic for(init; expr; update)
            else {
                slicedForNode = new ASNode(ASNode.Type.FOR);
                
                slicedForNode.setLineOfCode(ctx.getStart().getLine());
                
            	if ( nodeLevel=="block" ) {

            		ASNode forNode;
                    forNode = new ASNode(ASNode.Type.FOR);
                    forNode.setLineOfCode(ctx.getStart().getLine());
                    forNode.setProperty("is_sliced_root",true);
                    
                	ASNode  curentParentNode = slicedParentStack.peek();
                	if (curentParentNode != null && curentParentNode.getType() == ASNode.Type.BLOCK ) {
                		forNode.setType(ASNode.Type.NESTED_FOR);
                		slicedForNode.setType(ASNode.Type.NESTED_FOR);
                		AbstractSyntaxTree subTree = presentSubTreeStack.pop();
                		subTree.addVertex(forNode);
                		subTree.addEdge(slicedParentStack.peek(), forNode);
                		presentSubTreeStack.push(subTree);
                	}
                	else {
                        this.AST.addVertex(forNode);
                        this.AST.addEdge(parentStack.peek(), forNode);
                	}

                    slicedParentStack.push(forNode);
            	}
            	
//                slicedForNode = new ASNode(ASNode.Type.FOR);
                slicedForNode.setLineOfCode(ctx.getStart().getLine());
//                slicedForNode.setProperty("is_sliced_root",true);
                tree.addVertex(slicedForNode);
//                tree.addVertex(forNode);
                if(nodeLevel == "statement")
                	tree.addEdge(parentStack.peek(), slicedForNode);
                
                // for init
                if (ctx.forControl().forInit() != null) {
                    ASNode forInit = new ASNode(ASNode.Type.FOR_INIT);
                    tree.addVertex(forInit);
                    tree.addEdge(slicedForNode, forInit);
                    if (ctx.forControl().forInit().localVariableDeclaration() != null) {
                        slicedParentStack.push(forInit);
                        visit(ctx.forControl().forInit().localVariableDeclaration());
                        slicedParentStack.pop();
                    } else {
                        ASNode expr = new ASNode(ASNode.Type.STATEMENT);
                        expr.setCode(getOriginalCodeText(ctx.forControl().forInit().expressionList().expression(0)));
                        expr.setNormalizedCode(visit(ctx.forControl().forInit().expressionList().expression(0)));
                        expr.setLineOfCode(ctx.forControl().forInit().expressionList().expression(0).getStart().getLine());
                        tree.addVertex(expr);
                        tree.addEdge(forInit, expr);
                        //
                        int len = ctx.forControl().forInit().expressionList().expression().size();
                        for (int i = 1; i < len; ++i) {
                            expr = new ASNode(ASNode.Type.STATEMENT);
                            expr.setCode(getOriginalCodeText(ctx.forControl().forInit().expressionList().expression(i)));
                            expr.setNormalizedCode(visit(ctx.forControl().forInit().expressionList().expression(i)));
                            expr.setLineOfCode(ctx.forControl().forInit().expressionList().expression(i).getStart().getLine());
                            tree.addVertex(expr);
                            tree.addEdge(forInit, expr);
                        }
                    }
                }
                // for expr
                if (ctx.forControl().expression() != null) {
                    ASNode forExpr = new ASNode(ASNode.Type.CONDITION);
                    forExpr.setCode(getOriginalCodeText(ctx.forControl().expression()));
                    forExpr.setNormalizedCode(visit(ctx.forControl().expression()));
                    forExpr.setLineOfCode(ctx.forControl().expression().getStart().getLine());
                    tree.addVertex(forExpr);
                    tree.addEdge(slicedForNode, forExpr);
                }
                // for update
                if (ctx.forControl().forUpdate() != null) {
                    ASNode forUpdate = new ASNode(ASNode.Type.FOR_UPDATE);
                    tree.addVertex(forUpdate);
                    tree.addEdge(slicedForNode, forUpdate);
                    //
                    ASNode update = new ASNode(ASNode.Type.STATEMENT);
                    update.setCode(getOriginalCodeText(ctx.forControl().forUpdate().expressionList().expression(0)));
                    update.setNormalizedCode(visit(ctx.forControl().forUpdate().expressionList().expression(0)));
                    update.setLineOfCode(ctx.forControl().forUpdate().expressionList().expression(0).getStart().getLine());
                    tree.addVertex(update);
                    tree.addEdge(forUpdate, update);
                    //
                    int len = ctx.forControl().forUpdate().expressionList().expression().size();
                    for (int i = 1; i < len; ++i) {
                        update = new ASNode(ASNode.Type.STATEMENT);
                        update.setCode(getOriginalCodeText(ctx.forControl().forUpdate().expressionList().expression(i)));
                        update.setNormalizedCode(visit(ctx.forControl().forUpdate().expressionList().expression(i)));
                        update.setLineOfCode(ctx.forControl().forUpdate().expressionList().expression(i).getStart().getLine());
                        tree.addVertex(update);
                        tree.addEdge(forUpdate, update);
                    }
                }
            }
            //
            ASNode block = new ASNode(ASNode.Type.BLOCK);
            block.setLineOfCode(ctx.statement().getStart().getLine());
            tree.addVertex(block);
            tree.addEdge(slicedForNode, block);
//            parentStack.push(block);
            slicedParentStack.push(block);
            visit(ctx.statement());
            slicedParentStack.pop();
        }
        @Override
        public String visitForStatement(JavaParser.ForStatementContext ctx) {
            // 'for' '(' forControl ')' statement
            // forControl :  enhancedForControl  |  forInit? ';' expression? ';' forUpdate?
            // enhancedForControl :  variableModifier* typeType variableDeclaratorId ':' expression
            // forInit   :  localVariableDeclaration  |  expressionList
            // forUpdate :  expressionList
            //
            if (nodeLevel == "statement")
            {
            	processingForstatement(ctx, AST);
            	parentStack.pop();
            }            
            else if(nodeLevel == "block")
            {
            	initNestedBlock();
            	processingForstatement(ctx, presentSubTree);
//            	slicedAST.put(parentStack.peek(),presentSubTree);
            	if (presentSubTree.getAllEdges().size() >  0) 
            		slicedAST.put(slicedParentStack.pop(),presentSubTree);
            	initPresentSubTree();
//            	slicedAST.put(slicedParentStack.pop(),presentSubTree);
//            	initPresentSubTree();
            } 
            
            return "";
        }
        
        public void processWhileStatement(JavaParser.WhileStatementContext ctx,AbstractSyntaxTree tree )
        {
        	ASNode slicedWhileNode = new ASNode(ASNode.Type.WHILE);
        	if ( nodeLevel=="block" ) {
//            	tree = new AbstractSyntaxTree(presentSubTree.filePath, false);
        		ASNode whileNode = new ASNode(ASNode.Type.WHILE);
        		whileNode.setLineOfCode(ctx.getStart().getLine());
        		whileNode.setProperty("is_sliced_root",true);
        		
        		ASNode  curentParentNode = slicedParentStack.peek();
            	if (curentParentNode != null && curentParentNode.getType() == ASNode.Type.BLOCK) {
            		whileNode.setType(ASNode.Type.NESTED_WHILE);
            		slicedWhileNode.setType(ASNode.Type.NESTED_WHILE);
            		AbstractSyntaxTree subTree = presentSubTreeStack.pop();
            		subTree.addVertex(whileNode);
            		subTree.addEdge(slicedParentStack.peek(), whileNode);
            		presentSubTreeStack.push(subTree);
            	}
            	else {
	                AST.addVertex(whileNode);
	                AST.addEdge(parentStack.peek(), whileNode);
            	}
                slicedParentStack.push(whileNode);
        	}
        	 
        	 slicedWhileNode.setLineOfCode(ctx.getStart().getLine());
             tree.addVertex(slicedWhileNode);
             if(nodeLevel=="statement")
            	 tree.addEdge(parentStack.peek(), slicedWhileNode);
             //
             ASNode cond = new ASNode(ASNode.Type.CONDITION);
             cond.setCode(getOriginalCodeText(ctx.parExpression().expression()));
             cond.setNormalizedCode(visit(ctx.parExpression().expression()));
             cond.setLineOfCode(ctx.parExpression().expression().getStart().getLine());
             tree.addVertex(cond);
             tree.addEdge(slicedWhileNode, cond);
             //
             ASNode block = new ASNode(ASNode.Type.BLOCK);
             block.setLineOfCode(ctx.statement().getStart().getLine());
             tree.addVertex(block);
             tree.addEdge(slicedWhileNode, block);
             slicedParentStack.push(block);
             visit(ctx.statement());
             slicedParentStack.pop();
        }
        
        
        @Override
        public String visitWhileStatement(JavaParser.WhileStatementContext ctx) {
            // 'while' parExpression statement
            if (nodeLevel == "statement")
            {
            	processWhileStatement(ctx, AST);
            	parentStack.pop();
            }            
            else if(nodeLevel == "block")
            {	
            		initNestedBlock();
            	    processWhileStatement(ctx, presentSubTree);
                	if (presentSubTree.getAllEdges().size() >  0) 
                		slicedAST.put(slicedParentStack.pop(),presentSubTree);
                	initPresentSubTree();
//                    slicedAST.put(slicedParentStack.pop(),presentSubTree);
//                    initPresentSubTree();
//                    presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath,false);
            } 

            return "";
        }

        public void processDoWhileStatement(JavaParser.DoWhileStatementContext ctx, AbstractSyntaxTree tree)
        {
        	ASNode slicedDoWhileNode = new ASNode(ASNode.Type.DO_WHILE);
        	if ( nodeLevel=="block" ) {
//            	tree = new AbstractSyntaxTree(presentSubTree.filePath, false);
        		ASNode doWhileNode = new ASNode(ASNode.Type.DO_WHILE);
        		doWhileNode.setLineOfCode(ctx.getStart().getLine());
        		doWhileNode.setProperty("is_sliced_root",true);
        		
        		ASNode  curentParentNode = slicedParentStack.peek();
        		if (curentParentNode != null && curentParentNode.getType() == ASNode.Type.BLOCK) {
        			doWhileNode.setType(ASNode.Type.NESTED_DO_WHILE);
            		slicedDoWhileNode.setType(ASNode.Type.NESTED_DO_WHILE);
            		AbstractSyntaxTree subTree = presentSubTreeStack.pop();
            		subTree.addVertex(doWhileNode);
            		subTree.addEdge(slicedParentStack.peek(), doWhileNode);
            		presentSubTreeStack.push(subTree);
            	}
            	else {
	                AST.addVertex(doWhileNode);
	                AST.addEdge(parentStack.peek(), doWhileNode);
                }
                slicedParentStack.push(doWhileNode);
        	}
        	
            slicedDoWhileNode.setLineOfCode(ctx.getStart().getLine());
            tree.addVertex(slicedDoWhileNode);
            if(nodeLevel=="statement")
            	tree.addEdge(parentStack.peek(), slicedDoWhileNode);
            //
            ASNode cond = new ASNode(ASNode.Type.CONDITION);
            cond.setCode(getOriginalCodeText(ctx.parExpression().expression()));
            cond.setNormalizedCode(visit(ctx.parExpression().expression()));
            cond.setLineOfCode(ctx.parExpression().expression().getStart().getLine());
            tree.addVertex(cond);
            tree.addEdge(slicedDoWhileNode, cond);
            //
            ASNode block = new ASNode(ASNode.Type.BLOCK);
            block.setLineOfCode(ctx.statement().getStart().getLine());
            tree.addVertex(block);
            tree.addEdge(slicedDoWhileNode, block);
            slicedParentStack.push(block);
            visit(ctx.statement());
            slicedParentStack.pop();
        }

        @Override
        public String visitDoWhileStatement(JavaParser.DoWhileStatementContext ctx) {
            // 'do' statement 'while' parExpression ';'
            if (nodeLevel == "statement")
            {
            	processDoWhileStatement(ctx, AST);
            	parentStack.pop();
            }            
            else if(nodeLevel == "block")
            {	
            		initNestedBlock();
            	    processDoWhileStatement(ctx, presentSubTree);
                	if (presentSubTree.getAllEdges().size() >  0) 
                		slicedAST.put(slicedParentStack.pop(),presentSubTree);
                	initPresentSubTree();
//                    slicedAST.put(slicedParentStack.pop(),presentSubTree);
//                    initPresentSubTree();
//                    presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath,false);
            } 

            return "";
        }

        public void processTryStatement(JavaParser.TryStatementContext ctx, AbstractSyntaxTree tree)
        {
            // 'try' block (catchClause+ finallyBlock? | finallyBlock)
        	ASNode slicedTryNode = new ASNode(ASNode.Type.TRY);
        	if ( nodeLevel=="block" ) {
//            	tree = new AbstractSyntaxTree(presentSubTree.filePath, false);
        		ASNode tryNode = new ASNode(ASNode.Type.TRY);
        		tryNode.setLineOfCode(ctx.getStart().getLine());
        		tryNode.setProperty("is_sliced_root",true);
        		
        		   ASNode  curentParentNode = slicedParentStack.peek();
                   if (curentParentNode != null && curentParentNode.getType() == ASNode.Type.BLOCK) {
                	   tryNode.setType(ASNode.Type.NESTED_TRY);
	                   	slicedTryNode.setType(ASNode.Type.NESTED_TRY);
	               		AbstractSyntaxTree subTree = presentSubTreeStack.pop();
	               		subTree.addVertex(tryNode);
	               		subTree.addEdge(slicedParentStack.peek(),  tryNode);
	               		presentSubTreeStack.push(subTree);
               	}
               	else {
	                AST.addVertex(tryNode);
	                AST.addEdge(parentStack.peek(), tryNode);
               	}
                slicedParentStack.push(tryNode);
        	}
        	
            slicedTryNode.setLineOfCode(ctx.getStart().getLine());
            tree.addVertex(slicedTryNode);
            if(nodeLevel=="statement") 
            	tree.addEdge(parentStack.peek(), slicedTryNode);
            //
            ASNode tryBlock = new ASNode(ASNode.Type.BLOCK);
            tryBlock.setLineOfCode(ctx.block().getStart().getLine());
            tree.addVertex(tryBlock);
            tree.addEdge(slicedTryNode, tryBlock);
            slicedParentStack.push(tryBlock);
            visit(ctx.block());
            slicedParentStack.pop();
            // catchClause :  'catch' '(' variableModifier* catchType Identifier ')' block
            if (ctx.catchClause() != null && ctx.catchClause().size() > 0) {
                for (JavaParser.CatchClauseContext catchx : ctx.catchClause()) {
                    ASNode catchNode = new ASNode(ASNode.Type.CATCH);
                    catchNode.setLineOfCode(catchx.getStart().getLine());
                    tree.addVertex(catchNode);
                    tree.addEdge(slicedTryNode, catchNode);
                    //
                    ASNode catchType = new ASNode(ASNode.Type.TYPE);
                    catchType.setCode(catchx.catchType().getText());
                    catchType.setLineOfCode(catchx.catchType().getStart().getLine());
                    tree.addVertex(catchType);
                    tree.addEdge(catchNode, catchType);
                    //
                    ++varsCounter;
                    ASNode catchName = new ASNode(ASNode.Type.NAME);
                    String normalized = "$VARL_" + varsCounter;
                    vars.put(catchx.Identifier().getText(), normalized);
                    catchName.setCode(catchx.Identifier().getText());
                    catchName.setNormalizedCode(normalized);
                    catchName.setLineOfCode(catchx.getStart().getLine());
                    tree.addVertex(catchName);
                    tree.addEdge(catchNode, catchName);
                    //
                    ASNode catchBlock = new ASNode(ASNode.Type.BLOCK);
                    catchBlock.setLineOfCode(catchx.block().getStart().getLine());
                    tree.addVertex(catchBlock);
                    tree.addEdge(catchNode, catchBlock);
                    slicedParentStack.push(catchBlock);
                    visit(catchx.block());
                    slicedParentStack.pop();
                }
            }
            // finallyBlock :  'finally' block
            if (ctx.finallyBlock() != null) {
                ASNode finallyNode = new ASNode(ASNode.Type.FINALLY);
                finallyNode.setLineOfCode(ctx.finallyBlock().getStart().getLine());
                tree.addVertex(finallyNode);
                tree.addEdge(slicedTryNode, finallyNode);
                ASNode finallyBlockNode = new ASNode(ASNode.Type.BLOCK);
                finallyBlockNode.setLineOfCode(ctx.finallyBlock().getStart().getLine());
                tree.addVertex(finallyBlockNode);
                tree.addEdge(finallyNode,finallyBlockNode);
                slicedParentStack.push(finallyBlockNode);
                visit(ctx.finallyBlock().block());
                slicedParentStack.pop();
            }
        }
        @Override
        public String visitTryStatement(JavaParser.TryStatementContext ctx) {
            // 'try' block (catchClause+ finallyBlock? | finallyBlock)
            if (nodeLevel == "statement")
            {
            	processTryStatement(ctx, AST);
//            	parentStack.pop();
            }            
            else if(nodeLevel == "block")
            {	
            		initNestedBlock();
            	    processTryStatement(ctx, presentSubTree);
                	if (presentSubTree.getAllEdges().size() >  0) 
                		slicedAST.put(slicedParentStack.pop(),presentSubTree);
                	initPresentSubTree();
//                    slicedAST.put(slicedParentStack.pop(),presentSubTree);
//                    initPresentSubTree();
//                    presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath,false);
            } 
            return "";
        }

        public void processTryWithResourceStatement(JavaParser.TryWithResourceStatementContext ctx, AbstractSyntaxTree tree)
        {
        	ASNode slicedTryNode = new ASNode(ASNode.Type.TRY);
        	if ( nodeLevel=="block" ) {
//            	tree = new AbstractSyntaxTree(presentSubTree.filePath, false);
        		ASNode tryNode = new ASNode(ASNode.Type.TRY);
        		tryNode.setLineOfCode(ctx.getStart().getLine());
        		tryNode.setProperty("is_sliced_root",true);
        		
        		ASNode  curentParentNode = slicedParentStack.peek();
                if (curentParentNode != null && curentParentNode.getType() == ASNode.Type.BLOCK) {
             	   tryNode.setType(ASNode.Type.NESTED_TRY);
	                   	slicedTryNode.setType(ASNode.Type.NESTED_TRY);
	               		AbstractSyntaxTree subTree = presentSubTreeStack.pop();
	               		subTree.addVertex(tryNode);
	               		subTree.addEdge(slicedParentStack.peek(),  tryNode);
	               		presentSubTreeStack.push(subTree);
            	}
            	else {
	                AST.addVertex(tryNode);
	                AST.addEdge(parentStack.peek(), tryNode);
            	}
                slicedParentStack.push(tryNode);
        	}
        	
            slicedTryNode.setLineOfCode(ctx.getStart().getLine());
            tree.addVertex(slicedTryNode);
            if(nodeLevel=="statement")
            	tree.addEdge(parentStack.peek(), slicedTryNode);
            //
            ASNode resNode = new ASNode(ASNode.Type.RESOURCES);
            resNode.setLineOfCode(ctx.resourceSpecification().getStart().getLine());
            tree.addVertex(resNode);
            tree.addEdge(slicedTryNode, resNode);
            for (JavaParser.ResourceContext resctx : ctx.resourceSpecification().resources().resource()) {
                ASNode varNode = new ASNode(ASNode.Type.VARIABLE);
                varNode.setLineOfCode(resctx.getStart().getLine());
                tree.addVertex(varNode);
                tree.addEdge(resNode, varNode);
                //
                ASNode resType = new ASNode(ASNode.Type.TYPE);
                resType.setCode(resctx.classOrInterfaceType().getText());
                resType.setLineOfCode(resctx.classOrInterfaceType().getStart().getLine());
                tree.addVertex(resType);
                tree.addEdge(varNode, resType);
                //
                ++varsCounter;
                ASNode resName = new ASNode(ASNode.Type.NAME);
                String normalized = "$VARL_" + varsCounter;
                vars.put(resctx.variableDeclaratorId().Identifier().getText(), normalized);
                resName.setCode(resctx.variableDeclaratorId().getText());
                resName.setNormalizedCode(normalized);
                resName.setLineOfCode(resctx.variableDeclaratorId().getStart().getLine());
                tree.addVertex(resName);
                tree.addEdge(varNode, resName);
                //
                ASNode resInit = new ASNode(ASNode.Type.INIT_VALUE);
                resInit.setCode("= " + getOriginalCodeText(resctx.expression()));
                resInit.setNormalizedCode("= " + visit(resctx.expression()));
                resInit.setLineOfCode(resctx.expression().getStart().getLine());
                tree.addVertex(resInit);
                tree.addEdge(varNode, resInit);
            }
            ASNode tryBlock = new ASNode(ASNode.Type.BLOCK);
            tryBlock.setLineOfCode(ctx.block().getStart().getLine());
            tree.addVertex(tryBlock);
            tree.addEdge(slicedTryNode, tryBlock);
            slicedParentStack.push(tryBlock);
            visit(ctx.block());
            slicedParentStack.pop();
            //
            // catchClause :   'catch' '(' variableModifier* catchType Identifier ')' block
            if (ctx.catchClause().size() > 0 && ctx.catchClause() != null) {
                for (JavaParser.CatchClauseContext catchx : ctx.catchClause()) {
                    ASNode catchNode = new ASNode(ASNode.Type.CATCH);
                    catchNode.setLineOfCode(catchx.getStart().getLine());
                    tree.addVertex(catchNode);
                    tree.addEdge(slicedTryNode, catchNode);
                    //
                    ASNode catchType = new ASNode(ASNode.Type.TYPE);
                    catchType.setCode(catchx.catchType().getText());
                    catchType.setLineOfCode(catchx.catchType().getStart().getLine());
                    tree.addVertex(catchType);
                    tree.addEdge(catchNode, catchType);
                    //
                    ++varsCounter;
                    ASNode catchName = new ASNode(ASNode.Type.NAME);
                    String normalized = "$VARL_" + varsCounter;
                    vars.put(catchx.Identifier().getText(), normalized);
                    catchName.setCode(catchx.Identifier().getText());
                    catchName.setNormalizedCode(normalized);
                    catchName.setLineOfCode(catchx.catchType().getStart().getLine());
                    tree.addVertex(catchName);
                    tree.addEdge(catchNode, catchName);
                    //
                    ASNode catchBlock = new ASNode(ASNode.Type.BLOCK);
                    catchBlock.setLineOfCode(catchx.block().getStart().getLine());
                    tree.addVertex(catchBlock);
                    tree.addEdge(catchNode, catchBlock);
                    slicedParentStack.push(catchBlock);
                    visit(catchx.block());
                    slicedParentStack.pop();
                }
            }
            // finallyBlock :  'finally' block
            if (ctx.finallyBlock() != null) {
                ASNode finallyNode = new ASNode(ASNode.Type.FINALLY);
                finallyNode.setLineOfCode(ctx.finallyBlock().getStart().getLine());
                tree.addVertex(finallyNode);
                tree.addEdge(slicedTryNode, finallyNode);
                
                ASNode finallyBlockNode = new ASNode(ASNode.Type.BLOCK);
                finallyBlockNode.setLineOfCode(ctx.finallyBlock().getStart().getLine());
                tree.addVertex(finallyBlockNode);
                tree.addEdge(finallyNode,finallyBlockNode);
                
                slicedParentStack.push(finallyBlockNode);
                visit(ctx.finallyBlock().block());
                slicedParentStack.pop();
            }
        }
        @Override
        public String visitTryWithResourceStatement(JavaParser.TryWithResourceStatementContext ctx) {
            // 'try' resourceSpecification block catchClause* finallyBlock?
            // resourceSpecification :  '(' resources ';'? ')'
            // resources :  resource (';' resource)*
            // resource  :  variableModifier* classOrInterfaceType variableDeclaratorId '=' expression
            //
        	if (nodeLevel == "statement")
            {
            	processTryWithResourceStatement(ctx, AST);
//            	parentStack.pop();
            }            
            else if(nodeLevel == "block")
            {	
            		initNestedBlock();
            	    processTryWithResourceStatement(ctx, presentSubTree);
                	if (presentSubTree.getAllEdges().size() >  0) 
                		slicedAST.put(slicedParentStack.pop(),presentSubTree);
                	initPresentSubTree();
//                    slicedAST.put(slicedParentStack.pop(),presentSubTree);
//                    initPresentSubTree();
//                    presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath,false);
            } 
            return "";
        }

        public void processSwitchStatement(JavaParser.SwitchStatementContext ctx, AbstractSyntaxTree tree)
        {
        	ASNode slicedSwitchNode = new ASNode(ASNode.Type.SWITCH);
        	if ( nodeLevel=="block" ) {
//            	tree = new AbstractSyntaxTree(presentSubTree.filePath, false);
        		ASNode switchNode = new ASNode(ASNode.Type.SWITCH);
        		switchNode.setLineOfCode(ctx.getStart().getLine());
        		switchNode.setProperty("is_sliced_root",true);
        		
        		   ASNode  curentParentNode = slicedParentStack.peek();
                   if (curentParentNode != null && curentParentNode.getType() == ASNode.Type.BLOCK) {
                	switchNode.setType(ASNode.Type.NESTED_SWITCH);
                	slicedSwitchNode.setType(ASNode.Type.NESTED_SWITCH);
               		AbstractSyntaxTree subTree = presentSubTreeStack.pop();
               		subTree.addVertex( switchNode);
               		subTree.addEdge(slicedParentStack.peek(), switchNode);
               		presentSubTreeStack.push(subTree);
               	}
               	else {
	                AST.addVertex(switchNode);
	                AST.addEdge(parentStack.peek(), switchNode);
               	}
                slicedParentStack.push(switchNode);
        	}
        	
            slicedSwitchNode .setLineOfCode(ctx.getStart().getLine());
            tree.addVertex(slicedSwitchNode );
            if(nodeLevel=="statement")
            	tree.addEdge(parentStack.peek(), slicedSwitchNode );
            //
            ASNode varName = new ASNode(ASNode.Type.NAME);
            varName.setCode(getOriginalCodeText(ctx.parExpression().expression()));
            varName.setNormalizedCode(visit(ctx.parExpression().expression()));
            varName.setLineOfCode(ctx.parExpression().expression().getStart().getLine());
            tree.addVertex(varName);
            tree.addEdge(slicedSwitchNode , varName);
            //
            if (ctx.switchBlockStatementGroup() != null) {
                for (JavaParser.SwitchBlockStatementGroupContext grpx : ctx.switchBlockStatementGroup()) {
                    ASNode blockNode = new ASNode(ASNode.Type.BLOCK);
                    blockNode.setLineOfCode(grpx.blockStatement(0).getStart().getLine());
                    tree.addVertex(blockNode);
                    for (JavaParser.SwitchLabelContext lblctx : grpx.switchLabel())
                        visitSwitchLabel(lblctx, slicedSwitchNode , blockNode);
                    slicedParentStack.push(blockNode);
                    for (JavaParser.BlockStatementContext blk : grpx.blockStatement())
                        visit(blk);
                    slicedParentStack.pop();
                }
            }
            if (ctx.switchLabel() != null && ctx.switchLabel().size() > 0) {
                ASNode blockNode = new ASNode(ASNode.Type.BLOCK);
                blockNode.setLineOfCode(ctx.switchLabel(0).getStart().getLine());
                tree.addVertex(blockNode);
                for (JavaParser.SwitchLabelContext lblctx : ctx.switchLabel())
                    visitSwitchLabel(lblctx, slicedSwitchNode , blockNode);
            }
        }
        @Override
        public String visitSwitchStatement(JavaParser.SwitchStatementContext ctx) {
            // 'switch' parExpression '{' switchBlockStatementGroup* switchLabel* '}'
            // switchBlockStatementGroup  :   switchLabel+ blockStatement+
            // switchLabel :   'case' constantExpression ':'
            //             |   'case' enumConstantName ':'
            //             |   'default' ':'
            //
        	if (nodeLevel == "statement")
            {
            	processSwitchStatement(ctx, AST);
//            	parentStack.pop();
            }            
            else if(nodeLevel == "block")
            {	
            		initNestedBlock();
            	    processSwitchStatement(ctx, presentSubTree);
                	if (presentSubTree.getAllEdges().size() >  0) 
                		slicedAST.put(slicedParentStack.pop(),presentSubTree);
                	initPresentSubTree();
//                    slicedAST.put(slicedParentStack.pop(),presentSubTree);
//                    initPresentSubTree();
//                    presentSubTree = new AbstractSyntaxTree(presentSubTree.filePath,false);
            } 
            return "";
        }
        
        private void visitSwitchLabel(JavaParser.SwitchLabelContext lblctx, ASNode switchNode, ASNode blockNode) {
            if (nodeLevel=="statement")
            {
            	ASNode caseNode;
                if (lblctx.constantExpression() != null) {
                    caseNode = new ASNode(ASNode.Type.CASE);
                    caseNode.setCode(lblctx.constantExpression().getText());
                    caseNode.setLineOfCode(lblctx.getStart().getLine());
                } else if (lblctx.enumConstantName() != null) {
                    caseNode = new ASNode(ASNode.Type.CASE);
                    caseNode.setCode(lblctx.enumConstantName().getText());
                    caseNode.setLineOfCode(lblctx.getStart().getLine());
                } else {
                    caseNode = new ASNode(ASNode.Type.DEFAULT);
                    caseNode.setLineOfCode(lblctx.getStart().getLine());
                }
                AST.addVertex(caseNode);
                AST.addEdge(switchNode, caseNode);
                AST.addEdge(caseNode, blockNode);
            }
            else if(nodeLevel=="block") {
            	ASNode caseNode;
                if (lblctx.constantExpression() != null) {
                    caseNode = new ASNode(ASNode.Type.CASE);
                    caseNode.setCode(lblctx.constantExpression().getText());
                    caseNode.setLineOfCode(lblctx.getStart().getLine());
                } else if (lblctx.enumConstantName() != null) {
                    caseNode = new ASNode(ASNode.Type.CASE);
                    caseNode.setCode(lblctx.enumConstantName().getText());
                    caseNode.setLineOfCode(lblctx.getStart().getLine());
                } else {
                    caseNode = new ASNode(ASNode.Type.DEFAULT);
                    caseNode.setLineOfCode(lblctx.getStart().getLine());
                }
                presentSubTree.addVertex(caseNode);
                presentSubTree.addEdge(switchNode, caseNode);
                presentSubTree.addEdge(caseNode, blockNode);
            }
        }
        
        //=====================================================================//
        //                          EXPRESSIONS                                //
        //=====================================================================//
        
        @Override
        public String visitExprPrimary(JavaParser.ExprPrimaryContext ctx) {
            // exprPrimary :  primary
            //
            // primary :  '(' expression ')'
            //         |  'this'  |  'super'  |  literal  |  Identifier
            //         |   typeType '.' 'class'  |  'void' '.' 'class'
            //         |   nonWildcardTypeArguments (explicitGenericInvocationSuffix | 'this' arguments)
            //
            // nonWildcardTypeArguments :  '<' typeList '>'
            //
            // explicitGenericInvocationSuffix :  'super' superSuffix  |  Identifier arguments
            //
            JavaParser.PrimaryContext primary = ctx.primary();
            if (primary.expression() != null)
                return  " ( "  + visit(primary.expression()) +  " ) " ;
            if (primary.Identifier() != null)
                return normalizedIdentifier(primary.Identifier());
            if (primary.nonWildcardTypeArguments() != null) {
                if (primary.arguments() != null)
                    return getOriginalCodeText(primary.nonWildcardTypeArguments()) +  " this "  + visit(primary.arguments());
                else {
                    String suffix;
                    if (primary.explicitGenericInvocationSuffix().Identifier() != null)
                        suffix = normalizedIdentifier(primary.explicitGenericInvocationSuffix().Identifier())
                                + visit(primary.explicitGenericInvocationSuffix().arguments());
                    else
                        suffix =  " super "  + visit(primary.explicitGenericInvocationSuffix().superSuffix());
                    return getOriginalCodeText(primary.nonWildcardTypeArguments()) + suffix;
                }
            }
            return getOriginalCodeText(primary);
        }
        
        @Override
        public String visitExprDotID(JavaParser.ExprDotIDContext ctx) {
            // exprDotID :  expression '.' Identifier
            return visit(ctx.expression()) + " . " + normalizedIdentifier(ctx.Identifier());
        }
        
        @Override
        public String visitExprDotThis(JavaParser.ExprDotThisContext ctx) {
            // exprDotThis :  expression '.' 'this'
            return visit(ctx.expression()) + " .this ";
        }
        
        @Override
        public String visitExprDotNewInnerCreator(JavaParser.ExprDotNewInnerCreatorContext ctx) {
            // exprDotNewInnerCreator :  expression '.' 'new' nonWildcardTypeArguments? innerCreator
            //
            // innerCreator :  Identifier nonWildcardTypeArgumentsOrDiamond? classCreatorRest
            //
            // classCreatorRest :  arguments classBody?
            //
            return visit(ctx.expression()) + " .new " + getOriginalCodeText(ctx.nonWildcardTypeArguments())
                    + normalizedIdentifier(ctx.innerCreator().Identifier()) 
                    + getOriginalCodeText(ctx.innerCreator().nonWildcardTypeArgumentsOrDiamond())
                    + visit(ctx.innerCreator().classCreatorRest().arguments())
                    + visit(ctx.innerCreator().classCreatorRest().classBody());
        }
        
        @Override
        public String visitExprDotSuper(JavaParser.ExprDotSuperContext ctx) {
            // exprDotSuper :  expression '.' 'super' superSuffix
            return visit(ctx.expression()) + " .super " + visit(ctx.superSuffix());
        }
        
        @Override
        public String visitSuperSuffix(JavaParser.SuperSuffixContext ctx) {
            // superSuffix :  arguments  |  '.' Identifier arguments?
            String superSuffix = "";
            if (ctx.Identifier() != null)
                superSuffix = " . " + normalizedIdentifier(ctx.Identifier());
            if (ctx.arguments() != null)
                superSuffix += visit(ctx.arguments());
            return superSuffix;
        }
        
        @Override
        public String visitArguments(JavaParser.ArgumentsContext ctx) {
            // arguments :  '(' expressionList? ')'
            if (ctx.expressionList() == null)
                return " ( ) ";
            return " ( " + visit(ctx.expressionList()) + " ) ";
        }
        
        @Override
        public String visitExprDotGenInvok(JavaParser.ExprDotGenInvokContext ctx) {
            // exprDotGenInvok :  expression '.' explicitGenericInvocation
            //
            // explicitGenericInvocation :  nonWildcardTypeArguments explicitGenericInvocationSuffix
            //
            // nonWildcardTypeArguments :  '<' typeList '>'
            //
            // explicitGenericInvocationSuffix :  'super' superSuffix  |  Identifier arguments
            //
            String suffix;
            if (ctx.explicitGenericInvocation().explicitGenericInvocationSuffix().Identifier() != null)
                suffix = normalizedIdentifier(ctx.explicitGenericInvocation().explicitGenericInvocationSuffix().Identifier())
                        + visit(ctx.explicitGenericInvocation().explicitGenericInvocationSuffix().arguments());
            else
                suffix = " super " + visit(ctx.explicitGenericInvocation().explicitGenericInvocationSuffix().superSuffix());
            
            return visit(ctx.expression()) + " . " 
                    + getOriginalCodeText(ctx.explicitGenericInvocation().nonWildcardTypeArguments())
                    + suffix;
        }
        
        @Override
        public String visitExprArrayIndexing(JavaParser.ExprArrayIndexingContext ctx) {
            // exprArrayIndexing :  expression '[' expression ']'
            return visit(ctx.expression(0)) + " ["  + visit(ctx.expression(1)) +  " ] " ;
        }
        
        @Override
        public String visitExprMethodInvocation(JavaParser.ExprMethodInvocationContext ctx) { 
            // exprMethodInvocation :  expression '(' expressionList? ')'
            return visit(ctx.expression()) +  " ( "  + (ctx.expressionList() != null ? visit(ctx.expressionList()) : " ") + " ) ";
        }
        
        @Override
        public String visitExprNewCreator(JavaParser.ExprNewCreatorContext ctx) {
            // exprNewCreator :  'new' creator
            //
            // creator :  nonWildcardTypeArguments createdName classCreatorRest  
            //         |  createdName (arrayCreatorRest | classCreatorRest)
            //
            // createdName :  Identifier typeArgumentsOrDiamond? ('.' Identifier typeArgumentsOrDiamond?)*
            //             |   primitiveType
            //
            // classCreatorRest :  arguments classBody?
            //
            // arrayCreatorRest :  '[' (  ']' ('[' ']')* arrayInitializer 
            //                            | expression ']' ('[' expression ']')* ('[' ']')*   )
            //
            // arrayInitializer :  '{' (variableInitializer (',' variableInitializer)* (',')? )? '}'
            //
            // variableInitializer :  arrayInitializer  |  expression
            return visitChildren(ctx);
        }
        
        @Override
        public String visitExprCasting(JavaParser.ExprCastingContext ctx) {
            // exprCasting :  '(' typeType ')' expression
            return " ( " + getOriginalCodeText(ctx.typeType()) + " ) " + visit(ctx.expression());
        }
        
        @Override
        public String visitExprPostUnaryOp(JavaParser.ExprPostUnaryOpContext ctx) {
            // exprPostUnaryOp :  expression ('++' | '--')
            return visit(ctx.expression())  +  (ctx.getText().endsWith(" ++ ") ? " ++ " : " -- ");
        }
        
        @Override
        public String visitExprPreUnaryOp(JavaParser.ExprPreUnaryOpContext ctx) {
            // exprPreUnaryOp :  ('+'|'-'|'++'|'--') expression
            String op;
            if (ctx.getText().startsWith(" + "))
                op = ctx.getText().startsWith(" ++ ") ? " ++ " : " + ";
            else
                op = ctx.getText().startsWith(" -- ") ? " -- " : " - ";
            return op + visit(ctx.expression());
        }
        
        @Override
        public String visitExprNegation(JavaParser.ExprNegationContext ctx) {
            // exprNegation :  ('~'|'!') expression
            return (ctx.getText().startsWith(" ~ ") ? " ~ " : " ! ")  +  visit(ctx.expression());
        }
        
        @Override
        public String visitExprMulDivMod(JavaParser.ExprMulDivModContext ctx) {
            // exprMulDivMod :  expression ('*'|'/'|'%') expression
            char op = ctx.getText().substring(ctx.expression(0).getText().length()).charAt(0);
            return visit(ctx.expression(0)) + " " + op + " " + visit(ctx.expression(1));
        }
        
        @Override
        public String visitExprAddSub(JavaParser.ExprAddSubContext ctx) {
            // exprAddSub :  expression ('+'|'-') expression
            char op = ctx.getText().substring(ctx.expression(0).getText().length()).charAt(0);
            return visit(ctx.expression(0)) + " " + op + " " + visit(ctx.expression(1));
        }
        
        @Override
        public String visitExprBitShift(JavaParser.ExprBitShiftContext ctx) {
            // exprBitShift :  expression ('<' '<' | '>' '>' '>' | '>' '>') expression
            String sub = ctx.getText().substring(ctx.expression(0).getText().length());
            String op;
            if (sub.startsWith(" >>> "))
                op = " >>> ";
            else
                op = sub.substring(0, 2);
            return visit(ctx.expression(0)) + " " + op + " " + visit(ctx.expression(1));
        }
        
        @Override
        public String visitExprComparison(JavaParser.ExprComparisonContext ctx) {
            // exprComparison :  expression ('<=' | '>=' | '>' | '<') expression
            String sub = ctx.getText().substring(ctx.expression(0).getText().length());
            String op;
            if (sub.startsWith(" > ")) {
                if (sub.startsWith( " >= " ))
                    op =  "  >=  " ;
                else
                    op =  "  >  " ;
            } else {
                if (sub.startsWith( " <= " ))
                    op =  "  <=  " ;
                else
                    op =  "  <  " ;
            }
            return visit(ctx.expression(0)) + op + visit(ctx.expression(1));
        }
        
        @Override
        public String visitExprInstanceOf(JavaParser.ExprInstanceOfContext ctx) {
            // exprInstanceOf :  expression 'instanceof' typeType
            return visit(ctx.expression()) + " instanceof " + getOriginalCodeText(ctx.typeType());
        }
        
        @Override
        public String visitExprEquality(JavaParser.ExprEqualityContext ctx) {
            // exprEquality :  expression ('==' | '!=') expression
            String sub = ctx.getText().substring(ctx.expression(0).getText().length());
            String op;
            if (sub.startsWith( " == " ))
                op =  "  ==  " ;
            else
                op =  "  !=  " ;
            return visit(ctx.expression(0)) + op + visit(ctx.expression(1));
        }
        
        @Override
        public String visitExprBitAnd(JavaParser.ExprBitAndContext ctx) {
            // exprBitAnd :  expression '&' expression
            return visit(ctx.expression(0)) +  "  &  "  + visit(ctx.expression(1));
        }
        
        @Override
        public String visitExprBitXOR(JavaParser.ExprBitXORContext ctx) {
            // exprBitXOR :  expression '^' expression
            return visit(ctx.expression(0)) + " ^ " + visit(ctx.expression(1));
        }
        
        @Override
        public String visitExprBitOr(JavaParser.ExprBitOrContext ctx) {
            // exprBitOr :  expression '|' expression
            return visit(ctx.expression(0)) + " | " + visit(ctx.expression(1));
        }
        
        @Override
        public String visitExprLogicAnd(JavaParser.ExprLogicAndContext ctx) {
            // exprLogicAnd :  expression '&&' expression
            return visit(ctx.expression(0)) + " && " + visit(ctx.expression(1));
        }
        
        @Override
        public String visitExprLogicOr(JavaParser.ExprLogicOrContext ctx) {
            // exprLogicOr :  expression '||' expression
            return visit(ctx.expression(0)) + " || " + visit(ctx.expression(1));
        }
        
        @Override
        public String visitExprConditional(JavaParser.ExprConditionalContext ctx) {
            // exprConditional :  expression '?' expression ':' expression
            return visit(ctx.expression(0)) + " ? " + visit(ctx.expression(1)) + " : " + visit(ctx.expression(2));
        }
        
        @Override
        public String visitExprAssignment(JavaParser.ExprAssignmentContext ctx) {
            // exprAssignment :  expression  ( '='  | '+='  | '-='   | '*='  | '/=' | '&=' | 
            //                                 '|=' | '^=' | '>>=' | '>>>=' | '<<=' | '%=' )  expression
            return visit(ctx.expression(0)) + " ?= " + visit(ctx.expression(1));
        }
        
        @Override
        public String visitVariableInitializer(JavaParser.VariableInitializerContext ctx) {
            // variableInitializer :  arrayInitializer  |  expression
            if (ctx.expression() != null)
                return visit(ctx.expression());
            else
                return visit(ctx.arrayInitializer());
        }
        
        @Override
        public String visitArrayInitializer(JavaParser.ArrayInitializerContext ctx) {
            // arrayInitializer :  '{' (variableInitializer (',' variableInitializer)* (',')? )? '}'
            if (ctx.variableInitializer().size() > 0) {
                StringBuilder normalized = new StringBuilder( " { "  + visit(ctx.variableInitializer(0)));
                for (int i = 1; i < ctx.variableInitializer().size(); ++i)
                    normalized.append( " ,  " ).append(visit(ctx.variableInitializer(i)));
                return normalized.append( " } " ).toString();
            }
            return  " { } " ;
        }
        
        @Override
        public String visitExpressionList(JavaParser.ExpressionListContext ctx) {
            // expressionList :  expression (',' expression)*
            StringBuilder normalized = new StringBuilder(visit(ctx.expression(0)));
            for (int i = 1; i < ctx.expression().size(); ++i)
                normalized.append( " , ").append(visit(ctx.expression(i)));
            return normalized.toString();
        }

        //=====================================================================//
        //                          PRIVATE METHODS                            //
        //=====================================================================//

        /**
         * Get the original program text for the given parser-rule context. 
         * This is required for preserving white-spaces.
         */
        private String getOriginalCodeText(ParserRuleContext ctx) {
            if (ctx == null)
                return "";
            int start = ctx.start.getStartIndex();
            int stop = ctx.stop.getStopIndex();
//            FormattedText
            Interval interval = new Interval(start, stop);
            
//            CharStream stream = ctx.getInputStream();
//    		JavaLexer lexer = new JavaLexer(stream);
//    		CommonTokenStream   tokenStream  = new CommonTokenStream(lexer);
    		List<Token> ruleTokens = tokenStream.getTokens(ctx.start.getTokenIndex(), ctx.stop.getTokenIndex());
//            String text = " ";
//            for (int i = interval.a; i <= interval.b; i++) {
//              text += tokens.get(i).getText();
//              text += " ";
//            }
//            return text;
//            String s = new String(ctx.start.getInputStream()["data"],start,stop);
    		String text = " ";
    		for(Token item  : ruleTokens )
    		{
    			text += item.getText();
    			text += " ";
//    			text += ": ";
    		}
    		return text ;
//            return ctx.start.getInputStream().getText(interval);
        }
        
        private void resetLocalVars() {
            vars.clear();
            varsCounter = 0;
        }

        private String normalizedIdentifier(TerminalNode id) {
        	
            String normalized = vars.get(id.getText());
            if (normalized == null || normalized.isEmpty())
                normalized = fields.get(id.getText());
            if (normalized == null || normalized.isEmpty())
                normalized = methods.get(id.getText());
            if (normalized == null || normalized.isEmpty())
                normalized = id.getText();
            return normalized;
        }
    }
}
