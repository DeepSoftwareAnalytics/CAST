/*** In The Name of Allah ***/
package ghaffarian.progex.graphs.cfg;

import ghaffarian.collections.MatcherLinkedHashSet;
import ghaffarian.graphs.Edge;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ghaffarian.progex.utils.StringUtils;
import ghaffarian.nanologger.Logger;
import ghaffarian.progex.graphs.AbstractProgramGraph;
import java.io.IOException;
import java.util.Map.Entry;

/**
 * Control Flow Graph (CFG).
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class ControlFlowGraph extends AbstractProgramGraph<CFNode, CFEdge> {
	
	private String pkgName;
	public final String fileName;
	public String nodeLevel;
	private final List<CFNode> methodEntries;
	

	public ControlFlowGraph(String fileName) {
		super();
		this.pkgName = "";
		this.fileName = fileName;
		methodEntries = new ArrayList<>();
        properties.put("label", "CFG of " + fileName);
        properties.put("type", "Control Flow Graph (CFG)");
	}
	
	public void setPackage(String pkg) {
		pkgName = pkg;
	}
	
	public String getPackage() {
		return pkgName;
	}
	
	public void addMethodEntry(CFNode entry) {
		methodEntries.add(entry);
	}
	
	public CFNode[] getAllMethodEntries() {
		return methodEntries.toArray(new CFNode[methodEntries.size()]);
	}
	
    @Override
	public void exportDOT(String outDir) throws FileNotFoundException {
        if (!outDir.endsWith(File.separator))
            outDir += File.separator;
        File outDirFile = new File(outDir);
        outDirFile.mkdirs();
		String filename = fileName.substring(0, fileName.indexOf('.'));
		String filepath = outDir + filename + "-CFG.dot";
		
		try (PrintWriter dot = new PrintWriter(filepath, "UTF-8")) {
			if (nodeLevel == "statement") 
			{
				dot.println("digraph " + filename + "_CFG {");
	            dot.println("  // graph-vertices");
				Map<CFNode, String> nodeNames = new LinkedHashMap<>();
				int nodeCounter = 1;
				for (CFNode node: getAllVertices()) 
				{
					String name = "v" + nodeCounter++;
					nodeNames.put(node, name);
					StringBuilder label = new StringBuilder("  [label=\"");
					if (node.getLineOfCode() > 0)
						label.append(node.getLineOfCode()).append(":  ");
					label.append(StringUtils.escape(node.getCode())).append("\"];");
					dot.println("  " + name + label.toString());
				}
			
				dot.println("  // graph-edges");
				for (Edge<CFNode, CFEdge> edge: getAllEdges()) {
					String src = nodeNames.get(edge.source);
					String trg = nodeNames.get(edge.target);
					if (edge.label.type.equals(CFEdge.Type.EPSILON))
						dot.println("  " + src + " -> " + trg + ";");
					else
						dot.println("  " + src + " -> " + trg + "  [label=\"" + edge.label.type + "\"];");
				}
			}
			if (nodeLevel == "block") 
			{
				// merge the nodes in same line
				 Set<CFNode> allBlockVertices = new MatcherLinkedHashSet<>(16, VERTEX_MATCHER);
				 Set<Edge<CFNode,CFEdge>> allBlockEdges = new MatcherLinkedHashSet<>(32, EDGES_MATCHER);
				 int previousNodeLine = -100;
				 CFNode previousNode  = null;
				 StringBuilder codes = new StringBuilder();
				for (CFNode node: getAllVertices()) 
				{ 
					
					if (previousNode  == null)
					{
						allBlockVertices.add(node);
						previousNode  = node;
						previousNodeLine = node.getLineOfCode();
						codes = new StringBuilder();
						codes.append(previousNode.getCode()).append(";\n");
						continue;
					}
					if (node.getLineOfCode() !=  previousNodeLine) 
					{
						previousNode.setCode(codes.toString());
						allBlockVertices.add(previousNode);
						previousNode  = node;
						previousNodeLine = node.getLineOfCode();
						codes = new StringBuilder();
						codes.append(previousNode.getCode()).append(";\n");
						continue;
					}
					
					codes.append(node.getCode()).append(";\n");
					
				}
				previousNode.setCode(codes.toString());
				allBlockVertices.add(previousNode);
				
				dot.println("digraph " + filename + "_CFG {");
	            dot.println("  // graph-vertices");
				Map<Integer, String> nodeNames = new LinkedHashMap<>();
				int nodeCounter = 0;
				StringBuilder label = new StringBuilder("  ");
				String name = null;
				boolean preIsSimpleStmt = true;
				boolean isSimpleStmt = false;
				previousNodeLine = -100;
				for (CFNode node: allBlockVertices) 
				{   
					isSimpleStmt = node.geIsSimpleStmt("is_simple_stmt");
					boolean is_block = (isSimpleStmt == preIsSimpleStmt ) & (isSimpleStmt == true) &(previousNode.geIsEndBlock("is_end_block") == false);
//		
					if (!is_block ) {
						nodeNames.put(previousNodeLine, name);
						if (name != null)
							dot.println("  " + name + label.append("\"];").toString());
					
						name = "v" + nodeCounter++;
						label = new StringBuilder("  [label=\"");
						
						nodeNames.put(node.getLineOfCode(), name);
					}
//					name = "v" + nodeCounter;
					
					
					if (node.getLineOfCode() > 0)
						label.append(node.getLineOfCode()).append(":  ");
					label.append(StringUtils.escape(node.getCode())).append("\n");
					
					preIsSimpleStmt = isSimpleStmt;
					previousNodeLine = node.getLineOfCode();
					previousNode  = node;
				}
				nodeNames.put(previousNodeLine, name);
				dot.println("  " + name + label.append("\"];").toString());
				dot.println("  // graph-edges");
				for (Edge<CFNode, CFEdge> edge: getAllEdges()) {
					String src = nodeNames.get(edge.source.getLineOfCode());
					String trg = nodeNames.get(edge.target.getLineOfCode());
					if (src ==  null || trg == null || src == trg)
						continue;
					if (edge.label.type.equals(CFEdge.Type.EPSILON))
						dot.println("  " + src + " -> " + trg + ";");
					else
						dot.println("  " + src + " -> " + trg + "  [label=\"" + edge.label.type + "\"];");
				}
					
			}
			
			
			dot.println("  // end-of-graph\n}");
		} catch (UnsupportedEncodingException ex) {
			Logger.error(ex);
		}
		
		Logger.info("CFG exported to: " + filepath);
	}	

    @Override
    public void exportGML(String outDir) throws IOException {
        if (!outDir.endsWith(File.separator))
            outDir += File.separator;
        File outDirFile = new File(outDir);
        outDirFile.mkdirs();
		String filename = fileName.substring(0, fileName.indexOf('.'));
		String filepath = outDir + filename + "-CFG.gml";
		try (PrintWriter gml = new PrintWriter(filepath, "UTF-8")) {
			gml.println("graph [");
			gml.println("  directed 1");
			gml.println("  multigraph 1");
			for (Entry<String, String> property: properties.entrySet()) {
                switch (property.getKey()) {
                    case "directed":
                        continue;
                    default:
                        gml.println("  " + property.getKey() + " \"" + property.getValue() + "\"");
                }
            }
            gml.println("  file \"" + this.fileName + "\"");
            gml.println("  package \"" + this.pkgName + "\"\n");
            //
			Map<CFNode, Integer> nodeIDs = new LinkedHashMap<>();
			int nodeCounter = 0;
			for (CFNode node: getAllVertices()) {
				gml.println("  node [");
				gml.println("    id " + nodeCounter);
				gml.println("    line " + node.getLineOfCode());
				gml.println("    label \"" + StringUtils.escape(node.getCode()) + "\"");
				gml.println("  ]");
				nodeIDs.put(node, nodeCounter);
				++nodeCounter;
			}
            gml.println();
            //
			int edgeCounter = 0;
			for (Edge<CFNode, CFEdge> edge: getAllEdges()) {
				gml.println("  edge [");
				gml.println("    id " + edgeCounter);
				gml.println("    source " + nodeIDs.get(edge.source));
				gml.println("    target " + nodeIDs.get(edge.target));
				gml.println("    label \"" + edge.label.type + "\"");
				gml.println("  ]");
				++edgeCounter;
			}
			gml.println("]");
		} catch (UnsupportedEncodingException ex) {
			Logger.error(ex);
		}
		Logger.info("CFG exported to: " + filepath);
    }
	
    @Override
	public void exportJSON(String outDir) throws FileNotFoundException {
        if (!outDir.endsWith(File.separator))
            outDir += File.separator;
        File outDirFile = new File(outDir);
        outDirFile.mkdirs();
		String filename = fileName.substring(0, fileName.indexOf('.'));
		String filepath = outDir + filename + "-CFG.json";
		try (PrintWriter json = new PrintWriter(filepath, "UTF-8")) {
			json.println("{\n  \"directed\": true,");
			json.println("  \"multigraph\": true,");
			for (Entry<String, String> property: properties.entrySet()) {
                switch (property.getKey()) {
                    case "directed":
                        continue;
                    default:
                        json.println("  \"" + property.getKey() + "\": \"" + property.getValue() + "\",");
                }
            }
			json.println("  \"file\": \"" + fileName + "\",");
            json.println("  \"package\": \"" + this.pkgName + "\",\n");
            //
			json.println("  \"nodes\": [");
			Map<CFNode, Integer> nodeIDs = new LinkedHashMap<>();
			int nodeCounter = 0;
			for (CFNode node: getAllVertices()) {
                json.println("    {");
				json.println("      \"id\": " + nodeCounter + ",");
				json.println("      \"line\": " + node.getLineOfCode() + ",");
				json.println("      \"label\": \"" + StringUtils.escape(node.getCode()) + "\"");
				nodeIDs.put(node, nodeCounter);
				++nodeCounter;
                if (nodeCounter == getAllVertices().size())
                    json.println("    }");
                else
                    json.println("    },");
			}
            //
			json.println("  ],\n\n  \"edges\": [");
			int edgeCounter = 0;
			for (Edge<CFNode, CFEdge> edge: getAllEdges()) {
				json.println("    {");
				json.println("      \"id\": " + edgeCounter + ",");
				json.println("      \"source\": " + nodeIDs.get(edge.source) + ",");
				json.println("      \"target\": " + nodeIDs.get(edge.target) + ",");
				json.println("      \"label\": \"" + edge.label.type + "\"");
				++edgeCounter;
                if (edgeCounter == getAllEdges().size())
                    json.println("    }");
                else
                    json.println("    },");
			}
			json.println("  ]\n}");
		} catch (UnsupportedEncodingException ex) {
			Logger.error(ex);
		}
		Logger.info("CFG exported to: " + filepath);
	}
}
