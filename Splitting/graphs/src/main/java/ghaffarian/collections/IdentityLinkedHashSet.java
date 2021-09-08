/*** In The Name of Allah ***/
package ghaffarian.collections;

import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Hash table and linked list implementation of the <tt>Set</tt> interface,
 * with predictable iteration order.
 * 
 * This implementation differs from <tt>LinkedHashSet</tt> in that 
 * it uses reference-equality (or object-identity) in place of object-equality.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class IdentityLinkedHashSet<E> implements Set<E> {
    
    private final LinkedHashSet<IdentityWrapper> set;

    /**
     * Constructs a new, empty set with the default initial capacity (16) and load factor (0.8).
     */
    public IdentityLinkedHashSet() {
        set = new LinkedHashSet<>(16, 0.8f);
    }

    /**
     * Constructs a new, empty set with the given initial capacity and default load factor (0.8).
     */
    public IdentityLinkedHashSet(int capcity) {
        set = new LinkedHashSet<>(capcity, 0.8f);
    }
    
    /**
     * Copy constructor. 
     * Constructs a new set which contains all the elements in the given set.
     */
    public IdentityLinkedHashSet(Set<E> set) {
        this.set = new LinkedHashSet<>(16, 0.8f);
        for (E e: set)
            this.set.add(new IdentityWrapper(e));
    }
    
    @Override
    public boolean add(E e) {
        return set.add(new IdentityWrapper(e));
    }

    @Override
    public boolean addAll(Collection<? extends E> c) {
        boolean result = false;
        for (E e: c)
            result |= this.add(e);
        return result;
    }

    @Override
    public boolean remove(Object o) {
        return set.remove(new IdentityWrapper((E) o));
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        boolean result = false;
        for (Object obj: c)
            result |= this.remove((E) obj);
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
        return set.contains(new IdentityWrapper((E) obj));
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        boolean result = true;
        for (Object obj: c)
            result &= this.contains((E) obj);
        return result;
    }

    @Override
    public void clear() {
        set.clear();
    }
    
    @Override
    public Iterator<E> iterator() {
        Iterator<E> iter = new Iterator<E>() {
            private Iterator<IdentityWrapper> wrpIter = set.iterator();
            @Override
            public boolean hasNext() {
                return wrpIter.hasNext();
            }
            @Override
            public E next() {
                return (E) wrpIter.next().ELEM;
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
        for (IdentityWrapper wrp: set) {
            array[idx] = wrp.ELEM;
            ++idx;
        }
        return array;
    }

    @Override
    public <T> T[] toArray(T[] array) {
        int idx = 0;
        for (IdentityWrapper wrp: set) {
            array[idx] = (T) wrp.ELEM;
            ++idx;
        }
        return array;
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        boolean result = false;
        Iterator<IdentityWrapper> iter = set.iterator();
        while(iter.hasNext()) {
            IdentityWrapper wrp = iter.next();
            if (!c.contains(wrp.ELEM)) {
                iter.remove();
                result = true;
            }
        }
        return result;
    }

    /**
     * Wrapper class for objects in this collection,
     * which overrides equals and hashCode in a manner
     * that only concerns object identity.
     */
    private static class IdentityWrapper {
        
        public final Object ELEM;

        IdentityWrapper(Object elem) {
            this.ELEM = elem;
        }
        
        @Override
        public boolean equals(Object obj) {
            return (obj instanceof IdentityWrapper) && this.ELEM == ((IdentityWrapper) obj).ELEM;
        }
        
        @Override
        public int hashCode() {
            return System.identityHashCode(ELEM);
        }
    }
}
