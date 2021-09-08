/*** In The Name of Allah ***/
package ghaffarian.graphs;

/**
 * Interface used to match vertices or edges of graphs.
 * This class can be used to provide equals and hashCode logic 
 * for classes which do not implement equals and hashCode themselves.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public interface Matcher<T> {
    
    /**
     * Check whether the given objects are equal or not.
     */
    public boolean equals(T o1, T o2);
    
    /**
     * Calculate and return a hash value for the given object.
     */
    public int hashCode(T o);
    
}
