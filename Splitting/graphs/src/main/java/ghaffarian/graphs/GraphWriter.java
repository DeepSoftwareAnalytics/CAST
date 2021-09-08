/*** In The Name of Allah ***/
package ghaffarian.graphs;

import ghaffarian.graphs.utils.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Writes graph objects to various output formats.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class GraphWriter {
    
    /**
     * Writes a given graph object to a file in DOT format.
     * 
     * @param graph     the graph object to be written
     * @param filePath  the path of the file to write to
     */
    public static <V,E> void writeDOT(Graph<V,E> graph, String filePath) throws IOException {
        if (!filePath.toLowerCase().endsWith(".dot"))
            throw new IllegalArgumentException("File-path does not end with .dot suffix!");
		String filename = new File(filePath).getName();
		try (PrintWriter dot = new PrintWriter(filePath, "UTF-8")) {
            String edgeSymbol;
            String graphName = filename.substring(0, filename.lastIndexOf('.'));
            if (graph.isDirected()) {
                dot.println("digraph " + graphName + " {");
                edgeSymbol = " -> ";
            } else {
                dot.println("graph " + graphName + " {");
                edgeSymbol = " -- ";
            }
            dot.println("  // graph-vertices");
			Map<V, String> nodeNames = new LinkedHashMap<>();
			int nodeCounter = 1;
            Iterator<V> vertices = graph.allVerticesIterator();
			while (vertices.hasNext()) {
                V node = vertices.next();
				String name = "v" + nodeCounter++;
				nodeNames.put(node, name);
				StringBuilder label = new StringBuilder("  [label=\"");
				if (!node.toString().trim().isEmpty())
    				label.append(StringUtils.escape(node.toString()));
				dot.println("  " + name + label.append("\"];").toString());
			}
			dot.println("  // graph-edges");
            Iterator<Edge<V,E>> edges = graph.allEdgesIterator();
			while (edges.hasNext()) {
                Edge<V,E> edge = edges.next();
				String src = nodeNames.get(edge.source);
				String trg = nodeNames.get(edge.target);
				if (edge.label == null || edge.label.toString().trim().isEmpty())
					dot.println("  " + src + edgeSymbol + trg + ";");
				else
					dot.println("  " + src + edgeSymbol + trg + 
                            "  [label=\"" + StringUtils.escape(edge.label.toString()) + "\"];");
			}
			dot.println("  // end-of-graph\n}");
		}
    }
    
    /**
     * Writes a given graph object to a file in JSON format.
     * 
     * NOT IMPLEMENTED YET!
     * 
     * @param graph     the graph object to be written
     * @param filePath  the path of the file to write to
     */
    public static <V,E> void writeJSON(Graph<V,E> graph, String filePath) {
        throw new UnsupportedOperationException("Writing Graphs to JSON is NOT Implemented Yet!");
    }
    
}
