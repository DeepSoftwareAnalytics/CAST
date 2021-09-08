/*** In The Name of Allah ***/
package ghaffarian.graphs;

/**
 * A implementation of the Matcher interface for 
 * checking identity (reference equality) of objects.
 * 
 * According to this Matcher, two objects are considered equal 
 * if they are identical (they point to the same object in memory).
 * The hashCode method also uses System.identityHashCode(obj).
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class IdentityMatcher<T> implements Matcher<T> {

    @Override
    public boolean equals(T o1, T o2) {
        return o1 != null && o1 == o2;
    }

    @Override
    public int hashCode(T o) {
        return System.identityHashCode(o);
    }

}
