/*** In The Name of Allah ***/
package ghaffarian.graphs;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reads graph objects from various input sources.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class GraphReader {
    
    /**
     * Reads any DOT files inside the given directory and returns a list of graph objects.
     * 
     * @param dirPath path of the directory to read
     * @return        list of graphs generated from DOT files inside the given directory
     */
    public static List<Graph<String, String>> readDotDataset(String dirPath) throws IOException  {
        File dir = new File(dirPath);
        if (!dir.exists())
            throw new IllegalArgumentException("Path not found!");
        if (!dir.isDirectory()) 
            throw new IllegalArgumentException("Path is not a directory!");
        ArrayList<Graph<String, String>> graphDataset = new ArrayList<>(dir.list().length);
        for (File file: dir.listFiles()) {
            if (file.isFile() && file.getName().toLowerCase().endsWith(".dot"))
                graphDataset.add(readDOT(file.getPath()));
        }
        return graphDataset;
    }
    
    /**
     * Reads a DOT file and returns a single graph represented in the file.
     * Note that this method cannot read the general DOT language;
     * but only a small subset as written by the
     * {@link GraphWriter#writeDOT(ghaffarian.graphs.Graph, java.lang.String) 
     *  {@code GraphWriter.writeDOT(graph, path)}} method.
     * 
     * @param filePath  path of the DOT file to read
     * @return          graph object constructed from the given DOT file
     */
    public static Graph<String, String> readDOT(String filePath) throws IOException {
        if (!filePath.toLowerCase().endsWith(".dot"))
            throw new IllegalArgumentException("File-path does not end with .dot suffix!");
        Graph<String, String> graph;
		try (BufferedReader dot = new BufferedReader(new FileReader(filePath))) {
            // read graph type
            String line = dot.readLine().trim();
            if (line.startsWith("digraph"))
                graph = new Digraph();
            else
                graph = new UndiGraph();
            // skip any blank lines
            while (!(line = dot.readLine()).trim().equals("// graph-vertices")) { /* NOP! */ }
            // read graph vertices
			Map<String, String> vertexMap = new LinkedHashMap<>();
            while (!(line = dot.readLine()).trim().equals("// graph-edges")) {
                line = line.trim();
                String[] tokens = line.split("\\s+");
                int start = line.indexOf("[label=\"") + 8;
                int end = line.lastIndexOf("\"];");
                String vertex = line.substring(start, end);
                vertexMap.put(tokens[0], vertex);
                //System.out.println(tokens[0] + ":  " + vertex);
                graph.addVertex(vertex);
            }
            // read graph edges
            while (!(line = dot.readLine()).trim().equals("// end-of-graph")) {
                line = line.trim();
                String[] tokens = line.split("\\s+");
                if (tokens.length > 3) {
                    int start = line.indexOf("[label=\"") + 8;
                    int end = line.lastIndexOf("\"];");
                    String label = line.substring(start, end);
                    graph.addEdge(new Edge(vertexMap.get(tokens[0]), label, vertexMap.get(tokens[2])));
                    //System.out.println(tokens[0] + " -> " + tokens[2] + ":  " + edge);
                } else {
                    // remove semicolon
                    tokens[2] = tokens[2].substring(0, tokens[2].length() - 1);
                    graph.addEdge(new Edge(vertexMap.get(tokens[0]), null, vertexMap.get(tokens[2])));
                    //System.out.println(tokens[0] + " -> " + tokens[2]);
                }
            }
			return graph;
		}
    }
    
    /**
     * Reads a JSON file and returns a single graph represented in the file.
     * 
     * NOT IMPLEMENTED YET!
     * 
     * @param filePath  path of the JSON file to read
     * @return          graph object constructed from the given JSON file
     */
    public static <V,E> Graph<V,E> readJSON(String filePath) {
        throw new UnsupportedOperationException("Reading Graphs from JSON is NOT Implemented Yet!");
    }

}
