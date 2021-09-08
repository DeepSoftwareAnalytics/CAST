/*** In The Name of Allah ***/
package ghaffarian.graphs;

import java.util.Iterator;
import java.util.Set;

/**
 * A generic interface for labeled graphs.
 * 
 * @author ghaffarian
 */
public interface Graph<V,E> {
    
    /**
     * Check whether this graph is directed or not.
     */
    public boolean isDirected();
    
    /**
     * Add the given vertex to this graph.
     *
     * @return true if the vertex is added, or 
     *         false if such vertex is already in the graph.
     */
    public boolean addVertex(V v);

    /**
     * Remove the given vertex from this graph.
     *
     * @return true if the vertex is removed, or 
     *         false if no such vertex is in the graph.
     */
    public boolean removeVertex(V v);

    /**
     * Add the given edge to this graph. 
     * Both vertices (source and target) of the edge must be in the graph
     * otherwise, an exception is thrown indicating this issue.
     *
     * @return true if the edge is added, or 
     *         false if the edge is already in the graph.
     */
    public boolean addEdge(Edge<V, E> e);

    /**
     * Add an edge to this graph, connecting the given vertices. 
     * If a new edge can be added, the label of this new edge will be 'null'. 
     * Both vertices (source and target) of the edge must be in the graph
     * otherwise, an exception is thrown indicating this issue.
     *
     * @return true if a new edge is added, or 
     *         false if such an edge is already in the graph.
     */
    public boolean addEdge(V src, V trgt);

    /**
     * Remove the given edge from this graph.
     *
     * @return true if the vertex is removed, or
     *         false if no such vertex is in the graph.
     */
    public boolean removeEdge(Edge<V, E> e);

    /**
     * Remove all edges in this graph between the given source vertex and target vertex.
     *
     * @return the set of edges removed from this graph as a result of this operation.
     */
    public Set<Edge<V, E>> removeEdges(V src, V trgt);

    /**
     * Adds all vertices and edges of the given graph to this graph.
     *
     * @return true if this graph was modified; otherwise false.
     */
    public boolean addGraph(AbstractPropertyGraph<V, E> graph);

    /**
     * Return the number of vertices in this graph.
     */
    public int vertexCount();

    /**
     * Return the number of edges in this graph.
     */
    public int edgeCount();

    /**
     * Return a read-only iterator over all edges of the graph.
     */
    public Iterator<Edge<V, E>> allEdgesIterator();

    /**
     * Return a read-only iterator over all vertices of the graph.
     */
    public Iterator<V> allVerticesIterator();

    /**
     * Return a copy of the set of all edges in this graph.
     * This method has the overhead of creating of copy of the current set of edges.
     * Hence the returned collection is safe to use and modify (it is not linked to this graph).
     */
    public Set<Edge<V, E>> copyEdgeSet();

    /**
     * Return a copy of the set of all vertices in this graph.
     * This method has the overhead of creating of copy of the current set of vertices.
     * Hence the returned collection is safe to use and modify (it is not linked to this graph).
     */
    public Set<V> copyVertexSet();

    /**
     * Return a read-only iterator over the set of incoming edges to the given vertex.
     */
    public Iterator<Edge<V, E>> incomingEdgesIterator(V v);

    /**
     * Return a read-only iterator over the set of outgoing edges from the given vertex.
     */
    public Iterator<Edge<V, E>> outgoingEdgesIterator(V v);

    /**
     * Return a copy of the set of incoming edges to the given vertex.
     * This method has the overhead of creating of copy of the current set of incoming edges.
     * Hence the returned collection is safe to use and modify (it is not linked to this graph).
     */
    public Set<Edge<V, E>> copyIncomingEdges(V v);

    /**
     * Return a copy of the set of outgoing edges from the given vertex.
     * This method has the overhead of creating of copy of the current set of outgoing edges.
     * Hence the returned collection is safe to use and modify (it is not linked to this graph).
     */
    public Set<Edge<V, E>> copyOutgoingEdges(V v);

    /**
     * Return the count of incoming edges to the given vertex.
     */
    public int getInDegree(V v);

    /**
     * Return the count of outgoing edges from the given vertex.
     */
    public int getOutDegree(V v);

    /**
     * Return the set of edges with a label same as the given value.
     */
    public Set<Edge<V, E>> getEdgesWithLabel(E label);

    /**
     * Check if this graph contains the given edge.
     */
    public boolean containsEdge(Edge<V, E> e);

    /**
     * Check if this graph contains an edge between the given vertices.
     */
    public boolean containsEdge(V src, V trg);

    /**
     * Check if this graph contains the given vertex.
     */
    public boolean containsVertex(V v);

    /**
     * Check if this graph contains all edges in the given set.
     */
    public boolean containsAllEdges(Set<Edge<V, E>> vset);

    /**
     * Check if this graph contains all vertices in the given set.
     */
    public boolean containsAllVertices(Set<V> vset);

    /**
     * Check if this graph is a subgraph of the given base graph.
     */
    public boolean isSubgraphOf(Graph<V, E> base);

    /**
     * Check if this graph is a proper subgraph of the given base graph.
     * A proper subgraph is a subgraph which is not equal to the base graph.
     * A proper subgraph lacks at least one vertex or edge compared to the base.
     */
    public boolean isProperSubgraphOf(Graph<V, E> base);

    /**
     * Check whether this graph is connected or not.
     * Connectivity is determined by a breadth-first-traversal starting from a random vertex.
     */
    public boolean isConnected();
}
