/*** In The Name of Allah ***/
package ghaffarian.graphs;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * A generic abstract base class for labeled graphs; AKA property graphs.
 * In Graph-Theory, the mathematical definition of a graphs is 
 * a set of vertices and a set of edges where each edge connects two vertices.
 * A labeled graph or property graph includes some extra data (properties) associated the graph.
 * Nodes and edges in property graphs can have their own properties as well.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public abstract class AbstractPropertyGraph<V,E> implements Graph<V,E> {

	protected  Set<V> allVertices;
    protected Set<Edge<V,E>> allEdges;
    protected Map<V, Set<Edge<V,E>>> inEdges;
    protected Map<V, Set<Edge<V,E>>> outEdges;
    protected final Map<String, String> properties;
    
    /**
     * Default constructor for this abstract class.
     */
    public AbstractPropertyGraph() {
        properties = new HashMap<>();
    }
    
    /**
     * Copy constructor, which copies all properties of given graph.
     */
    public AbstractPropertyGraph(AbstractPropertyGraph graph) {
        properties = new HashMap<>(graph.properties);
    }
    
    /**
     * Check whether this graph has any property with given name.
     */
    public boolean hasProperty(String name) {
        return properties.containsKey(name);
    }
    
    /**
     * Returns the value of the given property name.
     * Returns null if no such property is set for this graph.
     */
    public String getProperty(String name) {
        return properties.get(name);
    }
    
    /**
     * Put given value with given name as a property for this graph.
     * 
     * @param name  name of the property to be set
     * @param value value of the property to be set
     * @return previous value for this property; null if there was no previous value
     */
    public String putProperty(String name, String value) {
        return properties.put(name, value);
    }
    
    /**
     * Remove the property with given name from this graph.
     * 
     * @param name name of the property to be removed
     * @return previous value of this property (null, if no value)
     */
    public String removeProperty(String name) {
        return properties.remove(name);
    }
    
    /**
     * Returns the <tt>Matcher</tt> object for matching vertices of this graph.
     */
    protected abstract Matcher<V> getVertexMatcher();
    
    /**
     * Returns the <tt>Matcher</tt> object for matching edges of this graph.
     */
    protected abstract Matcher<Edge<V,E>> getEdgesMatcher();
    
    @Override
    public int vertexCount() {
        return getAllVertices().size();
    }
    
    @Override
    public int edgeCount() {
        return getAllEdges().size();
    }
    
    @Override
    public Iterator<Edge<V,E>> allEdgesIterator() {
        return Collections.unmodifiableSet(getAllEdges()).iterator();
    }
    
    @Override
    public Iterator<V> allVerticesIterator() {
        return Collections.unmodifiableSet(getAllVertices()).iterator();
    }
    
    @Override
    public Iterator<Edge<V,E>> incomingEdgesIterator(V v) {
        return Collections.unmodifiableSet(inEdges.get(v)).iterator();
    }
    
    @Override
    public Iterator<Edge<V,E>> outgoingEdgesIterator(V v) {
        return Collections.unmodifiableSet(outEdges.get(v)).iterator();
    }
    
    @Override
    public boolean addGraph(AbstractPropertyGraph<V,E> graph) {
        boolean modified = false;
        for (V vrtx: graph.getAllVertices())
            modified |= addVertex(vrtx);
        for (Edge<V,E> edge: graph.getAllEdges())
            modified |= addEdge(edge);
        return modified;
    }
    
    @Override
    public int getInDegree(V v) {
        if (inEdges.get(v) == null)
            throw new IllegalArgumentException("No such vertex in this graph!");
        return inEdges.get(v).size();
    }
    
    @Override
    public int getOutDegree(V v) {
        if (outEdges.get(v) == null)
            throw new IllegalArgumentException("No such vertex in this graph!");
        return outEdges.get(v).size();
    }
    
    @Override
    public Set<Edge<V, E>> getEdgesWithLabel(E label) {
        Set<Edge<V, E>> edges = new LinkedHashSet<>();
        if (label == null) {
            for (Edge e: getAllEdges()) {
                if (e.label == null)
                    edges.add(e);
            }
        } else {
            for (Edge e: getAllEdges()) {
                if (label.equals(e.label))
                    edges.add(e);
            }
        }
        return edges;
    }
    
    @Override
    public boolean containsVertex(V v) {
        return getAllVertices().contains(v);
    }
    
    @Override
    public boolean containsAllEdges(Set<Edge<V, E>> set) {
        for (Edge<V, E> edge: set) {
            if (!containsEdge(edge))
                return false;
        }
        return true;
    }

    @Override
    public boolean containsAllVertices(Set<V> set) {
        for (V v: set) {
            if (!containsVertex(v))
                return false;
        }
        return true;
    }

    @Override
    public boolean isSubgraphOf(Graph<V,E> base) {
        if (isDirected() != base.isDirected())
            return false;
        if (this.vertexCount() > base.vertexCount() || this.edgeCount() > base.edgeCount())
            return false;
        if (base.containsAllVertices(this.getAllVertices())) {
            for (Edge<V,E> edge: this.getAllEdges())
                if (!base.containsEdge(edge))
                    return false;
            return true;
        } else
            return false;
    }
    
    @Override
    public boolean isProperSubgraphOf(Graph<V,E> base) {
        if (this.vertexCount() == base.vertexCount() && this.edgeCount() == base.edgeCount())
            return false;
        return isSubgraphOf(base);
    }
    
    @Override
    public boolean isConnected() {
        Set<V> visited = new HashSet<>();
        Deque<V> visiting = new ArrayDeque<>();
        visiting.add(getAllVertices().iterator().next());
        while (!visiting.isEmpty()) {
            V next = visiting.remove();
            visited.add(next);
            for (Edge<V,E> out: outEdges.get(next)) {
                if (!visited.contains(out.target))
                    visiting.add(out.target);
            }
            for (Edge<V,E> in: inEdges.get(next)) {
                if (!visited.contains(in.source))
                    visiting.add(in.source);
            }
        }
        return visited.size() == getAllVertices().size();
        // return visited.containsAll(allVertices);
    }

    /**
     * Returns a one-line string representation of this graph object.
     */
    public String toOneLineString() {
        StringBuilder str = new StringBuilder("{ ");
        for (V vrtx: getAllVertices()) {
            str.append(vrtx).append(": [ ");
            if (!outEdges.get(vrtx).isEmpty()) {
                for (Edge<V,E> edge: outEdges.get(vrtx)) {
                    if (edge.label == null)
                        str.append("(->").append(edge.target).append(") ");
                    else
                        str.append("(").append(edge.label).append("->").append(edge.target).append(") ");
                }
            }
            str.append("]; ");
        }
        str.append("}");
        return str.toString();
    }
    
    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        for (V vrtx: getAllVertices()) {
            str.append(vrtx).append(":\n");
            for (Edge<V,E> edge: outEdges.get(vrtx)) {
                if (edge.label == null)
                    str.append("  --> ").append(edge.target).append("\n");
                else
                    str.append("  --(").append(edge.label).append(")--> ").append(edge.target).append("\n");
            }
        }
        return str.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (!Objects.equals(getClass(), obj.getClass()))
            return false;
        final AbstractPropertyGraph<V,E> other = (AbstractPropertyGraph<V, E>) obj;
        return  this.isDirected() == other.isDirected() && 
                this.vertexCount() == other.vertexCount() && 
                this.edgeCount() == other.edgeCount() && 
                this.getAllVertices().containsAll(other.getAllVertices()) &&
                this.getAllEdges().containsAll(other.getAllEdges());
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 31 * hash + (isDirected()? 1 : 0);
        for (V v: getAllVertices())
            hash = 31 * hash + Objects.hashCode(v);
        for (Edge<V,E> e: getAllEdges())
            hash = 31 * hash + Objects.hashCode(e);
        return hash;
    }

	public Set<V> getAllVertices() {
		return allVertices;
	}

	public Set<Edge<V,E>> getAllEdges() {
		return allEdges;
	}

}
