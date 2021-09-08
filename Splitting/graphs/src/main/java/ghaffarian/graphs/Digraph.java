/*** In The Name of Allah ***/
package ghaffarian.graphs;

import ghaffarian.collections.MatcherLinkedHashMap;
import ghaffarian.collections.MatcherLinkedHashSet;
import java.util.Iterator;
import java.util.Set;

/**
 * A generic class for labeled digraphs (directed graphs),
 * also known as Property Graphs.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class Digraph<V, E> extends AbstractPropertyGraph<V, E> {
    
    protected final Matcher<V> VERTEX_MATCHER;
    protected final Matcher<Edge<V, E>> EDGES_MATCHER;
    
    /**
     * Construct a new empty Digraph object.
     * This instance will use a default matcher.
     */
    public Digraph() {
        super();
        EDGES_MATCHER = new DefaultMatcher<>();
        VERTEX_MATCHER = new DefaultMatcher<>();
        allEdges = new MatcherLinkedHashSet<>(32, EDGES_MATCHER);
        allVertices = new MatcherLinkedHashSet<>(16, VERTEX_MATCHER);
        inEdges = new MatcherLinkedHashMap<>(16, VERTEX_MATCHER);
        outEdges = new MatcherLinkedHashMap<>(16, VERTEX_MATCHER);
    }
    
    /**
     * Construct a new empty Digraph object, 
     * with the given <tt>Matcher</tt> objects for edges and vertices.
     */
    public Digraph(Matcher<V> vm, Matcher<Edge<V,E>> em) {
        super();
        properties.put("directed", "true");
        EDGES_MATCHER = em;
        VERTEX_MATCHER = vm;
        allEdges = new MatcherLinkedHashSet<>(32, EDGES_MATCHER);
        allVertices = new MatcherLinkedHashSet<>(16, VERTEX_MATCHER);
        inEdges = new MatcherLinkedHashMap<>(16, VERTEX_MATCHER);
        outEdges = new MatcherLinkedHashMap<>(16, VERTEX_MATCHER);
    }
    
    /**
     * Copy constructor.
     * Create a new Digraph instance by copying the state of the given graph object.
     * 
     * @param graph the Graph object to be copied
     */
    public Digraph(AbstractPropertyGraph<V,E> graph) {
        super(graph);
        properties.put("directed", "true");
        EDGES_MATCHER = graph.getEdgesMatcher();
        VERTEX_MATCHER = graph.getVertexMatcher();
        // copy all vertices and edges
        allEdges = new MatcherLinkedHashSet<>(graph.getAllEdges(), EDGES_MATCHER);
        allVertices = new MatcherLinkedHashSet<>(graph.getAllVertices(), VERTEX_MATCHER);
        // copy incoming-edges map
        inEdges = new MatcherLinkedHashMap<>(16, VERTEX_MATCHER);
        for (V v: graph.inEdges.keySet())
            inEdges.put(v, new MatcherLinkedHashSet<>(graph.inEdges.get(v), EDGES_MATCHER));
        // copy outgoing-edges map
        outEdges = new MatcherLinkedHashMap<>(16, VERTEX_MATCHER);
        for (V v: graph.outEdges.keySet())
            outEdges.put(v, new MatcherLinkedHashSet<>(graph.outEdges.get(v), EDGES_MATCHER));
    }
    
    @Override
    public boolean isDirected() {
        return true;
    }
    

    @Override
    protected Matcher<V> getVertexMatcher() {
        return VERTEX_MATCHER;
    }

    @Override
    protected Matcher<Edge<V, E>> getEdgesMatcher() {
        return EDGES_MATCHER;
    }

    @Override
    public boolean addVertex(V v) {
        if (getAllVertices().add(v)) {
            inEdges.put(v, new MatcherLinkedHashSet<>(8, EDGES_MATCHER));
            outEdges.put(v, new MatcherLinkedHashSet<>(8, EDGES_MATCHER));
            return true;
        }
        return false;
    }
    
    @Override
    public boolean removeVertex(V v) {
        if (getAllVertices().remove(v)) {
            getAllEdges().removeAll(inEdges.remove(v));
            getAllEdges().removeAll(outEdges.remove(v));
            return true;
        }
        return false;
    }
    
   
    public boolean justRemoveVertex(V v) {
        if (getAllVertices().remove(v)) {
            
            return true;
        }
        return false;
    }
    @Override
    public boolean addEdge(Edge<V,E> e) {
        if (!getAllVertices().contains(e.source))
            throw new IllegalArgumentException("No such source-vertex in this graph!");
        if (!getAllVertices().contains(e.target))
            throw new IllegalArgumentException("No such target-vertex in this graph!");
        if (getAllEdges().add(e)) {
            inEdges.get(e.target).add(e);
            outEdges.get(e.source).add(e);
            return true;
        }
        return false;
    }
    
    @Override
    public boolean addEdge(V src, V trgt) {
        return addEdge(new Edge<>(src, null, trgt));
    }
    
    @Override
    public boolean removeEdge(Edge<V,E> e) {
        if (getAllEdges().remove(e)) {
            inEdges.get(e.target).remove(e);
            outEdges.get(e.source).remove(e);
            return true;
        }
        return false;
    }
    
    @Override
    public Set<Edge<V,E>> removeEdges(V src, V trgt) {
        if (!getAllVertices().contains(src))
            throw new IllegalArgumentException("No such source-vertex in this graph!");
        if (!getAllVertices().contains(trgt))
            throw new IllegalArgumentException("No such target-vertex in this graph!");
        Set<Edge<V,E>> iterSet;
        Set<Edge<V,E>> removed = new MatcherLinkedHashSet<>(8, EDGES_MATCHER);
        if (inEdges.get(trgt).size() > outEdges.get(src).size()) {
            iterSet = outEdges.get(src);
            Iterator<Edge<V,E>> it = iterSet.iterator();
            while (it.hasNext()) {
                Edge<V,E> next = it.next();
                if (next.target.equals(trgt)) {
                    it.remove();
                    getAllEdges().remove(next);
                    inEdges.get(trgt).remove(next);
                    removed.add(next);
                }
            }
        } else {
            iterSet = inEdges.get(trgt);
            Iterator<Edge<V,E>> it = iterSet.iterator();
            while (it.hasNext()) {
                Edge<V,E> next = it.next();
                if (next.source.equals(src)) {
                    it.remove();
                    getAllEdges().remove(next);
                    outEdges.get(src).remove(next);
                    removed.add(next);
                }
            }
        }
        return removed;
    }
    
    @Override
    public Set<Edge<V,E>> copyEdgeSet() {
        return new MatcherLinkedHashSet<>(getAllEdges(), EDGES_MATCHER);
    }
    
    @Override
    public Set<V> copyVertexSet() {
        return new MatcherLinkedHashSet<>(getAllVertices(), VERTEX_MATCHER);
    }
    
    @Override
    public Set<Edge<V,E>> copyIncomingEdges(V v) {
        if (!getAllVertices().contains(v))
            throw new IllegalArgumentException("No such vertex in this graph!");
        return new MatcherLinkedHashSet<>(inEdges.get(v), EDGES_MATCHER);
    }
    
    @Override
    public Set<Edge<V,E>> copyOutgoingEdges(V v) {
        if (!getAllVertices().contains(v))
            throw new IllegalArgumentException("No such vertex in this graph!");
        return new MatcherLinkedHashSet<>(outEdges.get(v), EDGES_MATCHER);
    }
    
    @Override
    public boolean containsEdge(Edge<V,E> e) {
        return getAllEdges().contains(e);
    }
    
    @Override
    public boolean containsEdge(V src, V trg) {
        for (Edge<V,E> edge: outEdges.get(src)) {
            if (edge.target.equals(trg))
                return true;
        }
        return false;
    }
}
