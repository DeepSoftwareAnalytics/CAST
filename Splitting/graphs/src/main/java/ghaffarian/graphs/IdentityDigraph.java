/*** In The Name of Allah ***/
package ghaffarian.graphs;

import ghaffarian.collections.IdentityLinkedHashSet;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * A special kind of labeled digraph (directed graph).
 * While the mathematical definition of graphs only allows unique vertices and edges 
 * (a graph consists of a set of vertices and a set of edges), there are some applications
 * which might require to have equal but not identical vertices.
 * For such usecases, this class extends the Digraph implementation to allow
 * equal (but not identical) vertices and edges to be added to the graph.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class IdentityDigraph<V,E> extends Digraph<V,E> {
    
    // Two non-identity collections for efficient equality checking
    protected Set<V> allVerticesEq;
    protected Set<Edge<V,E>> allEdgesEq;

    /**
     * Construct a new empty Digraph object.
     */
    public IdentityDigraph() {
        allEdges = new IdentityLinkedHashSet<>(32);
        allVertices = new IdentityLinkedHashSet<>();
        inEdges = new IdentityHashMap<>();
        outEdges = new IdentityHashMap<>();
        //
        allEdgesEq = new LinkedHashSet<>(32);
        allVerticesEq = new LinkedHashSet<>();
    }
    
    /**
     * Copy constructor.
     * Create a new Digraph instance by copying the state of the given graph object.
     * 
     * @param graph the Graph object to be copied
     */
    public IdentityDigraph(AbstractPropertyGraph<V,E> graph) {
        // copy all vertices and edges
        allEdges = new IdentityLinkedHashSet<>(graph.getAllEdges());
        allVertices = new IdentityLinkedHashSet<>(graph.getAllVertices());
        if (graph instanceof IdentityDigraph) {
            IdentityDigraph<V, E> idgraph = (IdentityDigraph<V, E>) graph;
            allEdgesEq = new LinkedHashSet<>(idgraph.allEdgesEq);
            allVerticesEq = new LinkedHashSet<>(idgraph.allVerticesEq);
        }
        // copy incoming-edges map
        inEdges = new IdentityHashMap<>();
        for (V v : graph.inEdges.keySet())
            inEdges.put(v, new IdentityLinkedHashSet<>(graph.inEdges.get(v)));
        // copy outgoing-edges map
        outEdges = new IdentityHashMap<>();
        for (V v : graph.outEdges.keySet())
            outEdges.put(v, new IdentityLinkedHashSet<>(graph.outEdges.get(v)));
    }
    
    @Override
    public boolean addVertex(V v) {
        if (getAllVertices().add(v)) {
            inEdges.put(v, new IdentityLinkedHashSet<>());
            outEdges.put(v, new IdentityLinkedHashSet<>());
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
        Set<Edge<V,E>> removed = new IdentityLinkedHashSet<>();
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
        return new IdentityLinkedHashSet<>(getAllEdges());
    }
    
    @Override
    public Set<V> copyVertexSet() {
        return new IdentityLinkedHashSet<>(getAllVertices());
    }
    
    @Override
    public Set<Edge<V,E>> copyIncomingEdges(V v) {
        if (!getAllVertices().contains(v))
            throw new IllegalArgumentException("No such vertex in this graph!");
        return new IdentityLinkedHashSet<>(inEdges.get(v));
    }
    
    @Override
    public Set<Edge<V,E>> copyOutgoingEdges(V v) {
        if (!getAllVertices().contains(v))
            throw new IllegalArgumentException("No such vertex in this graph!");
        return new IdentityLinkedHashSet<>(outEdges.get(v));
    }
    
    /**
     * {@inheritDoc}
     * 
     * Note that this method uses equality and not identity; 
     * which means if this method returns true, 
     * the given object isn't necessarily included in this {@code IdentityGraph}.
     * 
     * If identity checking is desired, use {@link #hasVertex(java.lang.Object) {@code hasVertex(v)}}.
     * 
     * @return  {@code true} if the input vertex equals to some vertex in this graph;
     *          {@code false} otherwise.
     */
    @Override
    public boolean containsVertex(V v) {
        return allVerticesEq.contains(v);
    }
    
    /**
     * Checks if the given vertex is included in this graph.
     * Note that this method uses identity checking, and not equality.
     * 
     * If equality checking is desired, use {@link #containsVertex(java.lang.Object) {@code containsVertex(v)}}.
     * 
     * @return  {@code true} if the vertex object is included in this graph;
     *          {@code false} otherwise.
     */
    public boolean hasVertex(V v) {
        return getAllVertices().contains(v);
    }
    
    /**
     * {@inheritDoc}
     * 
     * Note that this method uses equality and not identity; 
     * which means if this method returns {@code true}, 
     * the given object isn't necessarily included in this {@code IdentityGraph}.
     * 
     * If identity checking is desired, use {@link #hasEdge(ghaffarian.graphs.Edge) {@code hasEdge(e)}}.
     * 
     * @return  {@code true} if the input edge equals to some edge in this graph;
     *          {@code false} otherwise.
     */
    @Override
    public boolean containsEdge(Edge<V,E> e) {
        return allEdgesEq.contains(e);
    }
    
    /**
     * Checks if the given edge is included in this graph.
     * Note that this method uses identity checking, and not equality.
     * 
     * If equality checking is desired, use {@link #containsEdge(ghaffarian.graphs.Edge) {@code containsEdge(e)}}.
     * 
     * @return  {@code true} if the edge object is included in this graph;
     *          {@code false} otherwise.
     */
    public boolean hasEdge(Edge<V,E> e) {
        return getAllEdges().contains(e);
    }
    
    /**
     * {@inheritDoc}
     * 
     * Note that this method uses equality and not identity; 
     * which means if this method returns {@code true}, 
     * the given objects are not necessarily included in this {@code IdentityGraph}.
     * 
     * If identity checking is desired, use {@link #hasEdge(ghaffarian.graphs.Edge) {@code hasEdge(src, trg)}}.
     * 
     * @return  {@code true} if some edge in this graph connects two vertices equal to the inputs;
     *          {@code false} otherwise.
     */
    @Override
    public boolean containsEdge(V src, V trg) {
        if (!allVerticesEq.contains(src) || !allVerticesEq.contains(trg))
            return false;
        for (V v: outEdges.keySet()) {
            if (v.equals(src)) {
                for (Edge<V,E> edge: outEdges.get(v))
                    if (edge.target.equals(trg))
                        return true;
            }
        }
        return false;
    }
    
    /**
     * Checks if an edge connects the given vertices in this graph.
     * Note that this method uses identity checking, and not equality.
     * 
     * If equality checking is desired, 
     * use {@link #containsEdge(java.lang.Object, java.lang.Object) {@code containsEdge(src, trg)}}.
     * 
     * @return  {@code true} if the two vertex objects are connected in this graph;
     *          {@code false} otherwise.
     */
    public boolean hasEdge(V src, V trg) {
        for (Edge<V, E> edge: outEdges.get(src)) {
            if (edge.target == trg)
                return true;
        }
        return false;
    }
}
