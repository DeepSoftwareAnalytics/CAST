/*** In The Name of Allah ***/
package ghaffarian.graphs;

import java.io.IOException;

import static org.junit.Assert.*;
import org.junit.*;

/**
 * Testing graph operations.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class GraphsTests {
    
    @Test
    public void dotWriterTest() throws IOException {
        Graph<String, String> graph = new Digraph<>();
        // add vertices
        String  v1 = "V1-test",
                v2 = "V2-test",
                v3 = "V3-test",
                v4 = "V4-test";
        graph.addVertex(v1);
        graph.addVertex(v2);
        graph.addVertex(v3);
        graph.addVertex(v4);
        // add edges
        String  e1 = "E1-test",
                //e2 = "E2-test",
                //e3 = "E3-test",
                //e4 = "E4-test",
                //e5 = "E5-test",
                //e6 = "E6-test",
                e7 = "E7-test";
        graph.addEdge(new Edge(v1, e1, v2));
        graph.addEdge(new Edge(v1, e7, v3));
        graph.addEdge(new Edge(v1, e1, v4));
        graph.addEdge(new Edge(v2, e7, v3));
        graph.addEdge(new Edge(v2, e1, v4));
        graph.addEdge(new Edge(v3, e7, v4));
        graph.addEdge(new Edge(v4, e7, v1));
        // write to file
        GraphWriter.writeDOT(graph, "src/test/resources/write_test_1.dot");
        // assertion
        assertTrue(true);
    }
    
    @Test
    public void dotReaderTest() throws IOException {
        // read graph from DOT file
        Graph<String, String> graph = GraphReader.readDOT("src/test/resources/read_test_1.dot");
        // check directed/undirected property
        assertTrue(graph.isDirected());
        // vertices
        String  v1 = "V1-test",
                v2 = "V2-test",
                v3 = "V3-test",
                v4 = "V4-test";
        // edges
        String  e1 = "E1-test",
                //e2 = "E2-test",
                //e3 = "E3-test",
                //e4 = "E4-test",
                //e5 = "E5-test",
                //e6 = "E6-test",
                e7 = "E7-test";
        // check edges
        assertTrue(graph.containsEdge(new Edge(v1, e1, v2)));
        assertTrue(graph.containsEdge(new Edge(v1, e7, v3)));
        assertTrue(graph.containsEdge(new Edge(v1, e1, v4)));
        assertTrue(graph.containsEdge(new Edge(v2, e7, v3)));
        assertTrue(graph.containsEdge(new Edge(v2, e1, v4)));
        assertTrue(graph.containsEdge(new Edge(v3, e7, v4)));
        assertTrue(graph.containsEdge(new Edge(v4, e7, v1)));
    }
    
    @Test
    public void subGraphTest() throws IOException {
        // read graphs from DOT file
        Graph<String, String> cfg = GraphReader.readDOT("src/test/resources/CFG.dot");
        Graph<String, String> sub = GraphReader.readDOT("src/test/resources/sub-CFG.dot");
        // assertion
        assertTrue(sub.isSubgraphOf(cfg));
        assertFalse(cfg.isSubgraphOf(sub));
    }
}
