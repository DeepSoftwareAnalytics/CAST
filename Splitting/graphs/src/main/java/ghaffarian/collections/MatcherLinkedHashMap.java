/*** In The Name of Allah ***/
package ghaffarian.collections;

import ghaffarian.graphs.DefaultMatcher;
import ghaffarian.graphs.Holder;
import ghaffarian.graphs.Matcher;
import java.util.AbstractMap;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * Hash table and linked list implementation of the <tt>Map</tt> interface, 
 * with predictable iteration order.
 * 
 * This implementation differs from <tt>LinkedHashSet</tt> in that it uses 
 * a given <tt>Matcher</tt> object to test equality and calculate hash values.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class MatcherLinkedHashMap<K, V> implements Map<K, V> {
    
    public final Matcher<K> matcher;
    private final LinkedHashMap<Holder<K>, V> map;
    
    /**
     * Constructs a new, empty set with the default initial capacity (16) and a default matcher.
     */
    public MatcherLinkedHashMap() {
        this(16, new DefaultMatcher<K>());
    }

    /**
     * Constructs a new, empty map with the given initial capacity and given matcher object.
     */
    public MatcherLinkedHashMap(int capacity, Matcher<K> matcher) {
        map = new LinkedHashMap<>(capacity, 0.8f);
        this.matcher = matcher;
    }

    @Override
    public V put(K key, V value) {
        return map.put(new Holder<>(key, matcher), value);
    }

    @Override
    public void putAll(Map<? extends K, ? extends V> m) {
        for (K key: m.keySet())
            map.put(new Holder<>(key, matcher), m.get(key));
    }

    @Override
    public V get(Object key) {
        return map.get(new Holder<>((K) key, matcher));
    }

    @Override
    public V remove(Object key) {
        return map.remove(new Holder<>((K) key, matcher));
    }

    @Override
    public void clear() {
        map.clear();
    }

    @Override
    public int size() {
        return map.size();
    }

    @Override
    public boolean isEmpty() {
        return map.isEmpty();
    }

    @Override
    public boolean containsKey(Object key) {
        return map.containsKey(new Holder<>((K) key, matcher));
    }

    @Override
    public boolean containsValue(Object value) {
        return map.containsValue(value);
    }

    @Override
    public Set<K> keySet() {
        Set<K> set = new MatcherLinkedHashSet<>(map.size(), matcher);
        for (Holder<K> k: map.keySet())
            set.add(k.object);
        return set;
    }

    @Override
    public Collection<V> values() {
        return map.values();
    }

    @Override
    public Set<Map.Entry<K, V>> entrySet() {
        Set<Map.Entry<K, V>> entries = new LinkedHashSet<>(map.size());
        for (Holder<K> k: map.keySet())
            entries.add(new AbstractMap.SimpleImmutableEntry<>(k.object, map.get(k)));
        return entries;
    }
}
