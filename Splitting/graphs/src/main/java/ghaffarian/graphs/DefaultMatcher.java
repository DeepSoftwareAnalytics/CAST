/*** In The Name of Allah ***/
package ghaffarian.graphs;

/**
 * A default implementation for the Matcher interface 
 * which uses the objects own equals and hashCode methods.
 * 
 * This implementation is suitable for classes where 
 * equals and hashCode methods are implemented;
 * otherwise, a custom Matcher must be implemented.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class DefaultMatcher<T> implements Matcher<T> {

    @Override
    public boolean equals(T o1, T o2) {
        if (o1 == null || o2 == null)
            return false;
        return o1.equals(o2) && o2.equals(o1);
    }

    @Override
    public int hashCode(T o) {
        return o.hashCode();
    }

}
