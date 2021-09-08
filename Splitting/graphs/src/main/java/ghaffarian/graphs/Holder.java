/*** In The Name of Allah ***/
package ghaffarian.graphs;

/**
 * A generic object holder class.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class Holder<T> {
    
    public final T object;
    public final Matcher<T> matcher;
    
    public Holder(T obj, Matcher<T> mchr) {
        object = obj;
        matcher = mchr;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (obj == null)
            return false;
        if (object == obj)
            return true;
        if (this.getClass() != obj.getClass())
            return false;
        return matcher.equals(object, ((Holder<T>) obj).object);
    }

    @Override
    public int hashCode() {
        return matcher.hashCode(object);
    }

}
