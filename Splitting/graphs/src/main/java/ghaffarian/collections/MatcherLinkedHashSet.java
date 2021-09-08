/*** In The Name of Allah ***/
package ghaffarian.collections;

import ghaffarian.graphs.DefaultMatcher;
import ghaffarian.graphs.Holder;
import ghaffarian.graphs.Matcher;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Hash table and linked list implementation of the <tt>Set</tt> interface,
 * with predictable iteration order.
 * 
 * This implementation differs from <tt>LinkedHashSet</tt> in that it uses 
 * a given <tt>Matcher</tt> object to test equality and calculate hash values.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class MatcherLinkedHashSet<T> implements Set<T> {
    
    protected final Matcher<T> matcher;
    private final LinkedHashSet<Holder<T>> set;

    /**
     * Constructs a new, empty set with the default initial capacity (16) and load factor (0.8).
     */
    public MatcherLinkedHashSet() {
        this(16);
    }

    /**
     * Constructs a new, empty set with the given initial capacity and default load factor (0.8).
     */
    public MatcherLinkedHashSet(int capacity) {
        this(capacity, new DefaultMatcher<>());
    }
    
    /**
     * Constructs a new, empty set with the given initial capacity and given matcher object.
     */
    public MatcherLinkedHashSet(int capacity, Matcher<T> matcher) {
        set = new LinkedHashSet<>(capacity, 0.8f);
        this.matcher = matcher;
    }
    
    /**
     * Constructs a new set which contains all the elements in the given set.
     */
    public MatcherLinkedHashSet(Set<T> set, Matcher<T> matcher) {
        this.matcher = matcher;
        this.set = new LinkedHashSet<>(16, 0.8f);
        for (T e: set)
            this.set.add(new Holder<>(e, matcher));
    }
    
    @Override
    public boolean add(T e) {
        return set.add(new Holder<>(e, matcher));
    }

    @Override
    public boolean addAll(Collection<? extends T> c) {
        boolean result = false;
        for (T e: c)
            result |= this.add(e);
        return result;
    }

    @Override
    public boolean remove(Object o) {
        return set.remove(new Holder((T) o, matcher));
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        boolean result = false;
        for (Object obj: c)
            result |= this.remove((T) obj);
        return result;
    }

    @Override
    public int size() {
        return set.size();
    }

    @Override
    public boolean isEmpty() {
        return set.isEmpty();
    }

    @Override
    public boolean contains(Object obj) {
        return set.contains(new Holder((T) obj, matcher));
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        boolean result = true;
        for (Object obj: c)
            result &= this.contains((T) obj);
        return result;
    }

    @Override
    public void clear() {
        set.clear();
    }
    
    @Override
    public Iterator<T> iterator() {
        Iterator<T> iter = new Iterator<T>() {
            private Iterator<Holder<T>> wrpIter = set.iterator();
            @Override
            public boolean hasNext() {
                return wrpIter.hasNext();
            }
            @Override
            public T next() {
                return (T) wrpIter.next().object;
            }
            @Override
            public void remove() {
                wrpIter.remove();
            }
        };
        return iter;
    }

    @Override
    public Object[] toArray() {
        Object[] array = new Object[set.size()];
        int idx = 0;
        for (Holder wrp: set) {
            array[idx] = wrp.object;
            ++idx;
        }
        return array;
    }

    @Override
    public <T> T[] toArray(T[] array) {
        int idx = 0;
        for (Holder wrp: set) {
            array[idx] = (T) wrp.object;
            ++idx;
        }
        return array;
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        boolean result = false;
        Iterator<Holder<T>> iter = set.iterator();
        while(iter.hasNext()) {
            Holder wrp = iter.next();
            if (!c.contains(wrp.object)) {
                iter.remove();
                result = true;
            }
        }
        return result;
    }
}
