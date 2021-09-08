/*** In The Name of Allah ***/
package ghaffarian.graphs;

import ghaffarian.collections.IdentityLinkedHashSet;
import ghaffarian.collections.MatcherLinkedHashSet;
import java.util.Set;

import static org.junit.Assert.*;
import org.junit.*;

/**
 * Testing collections.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class CollectionsTests {
    
    @Test
    public void identitySetTest()  {
        Atom hydrogen1 = new Atom("Hydrogen", "H");
        Atom hydrogen2 = new Atom("Hydrogen", "H");
        Atom oxygen = new Atom("Oxygen", "O");
        
        Set<Atom> identitySet = new IdentityLinkedHashSet<>(); 
        //java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        
        assertTrue(identitySet.add(oxygen));
        assertFalse(identitySet.add(oxygen));
        assertTrue(identitySet.contains(oxygen));
        //
        assertTrue(identitySet.add(hydrogen1));
        assertFalse(identitySet.add(hydrogen1));
        assertTrue(identitySet.contains(hydrogen1));
        assertFalse(identitySet.contains(hydrogen2));
        //
        assertTrue(identitySet.add(hydrogen2));
        assertFalse(identitySet.add(hydrogen2));
        assertTrue(identitySet.contains(hydrogen2));
        assertEquals(3, identitySet.size());
    }
    
    @Test
    public void normalMatcherSetTest() {
        Atom hydrogen1 = new Atom("Hydrogen", "H");
        Atom hydrogen2 = new Atom("Hydrogen", "H");
        Atom oxygen = new Atom("Oxygen", "O");
        
        Set<Atom> matcherSet = new MatcherLinkedHashSet<>(); 
        
        assertTrue(matcherSet.add(oxygen));
        assertFalse(matcherSet.add(oxygen));
        assertTrue(matcherSet.contains(oxygen));
        //
        assertTrue(matcherSet.add(hydrogen1));
        assertFalse(matcherSet.add(hydrogen1));
        assertFalse(matcherSet.add(hydrogen2));
        assertTrue(matcherSet.contains(hydrogen1));
        assertTrue(matcherSet.contains(hydrogen2));
        //
        assertEquals(2, matcherSet.size());
    }
    
    @Test
    public void identityMatcherSetTest() {
        Atom hydrogen1 = new Atom("Hydrogen", "H");
        Atom hydrogen2 = new Atom("Hydrogen", "H");
        Atom oxygen = new Atom("Oxygen", "O");
        
        Set<Atom> matcherSet = new MatcherLinkedHashSet<>(8, new IdentityMatcher<>()); 
        
        assertTrue(matcherSet.add(oxygen));
        assertFalse(matcherSet.add(oxygen));
        assertTrue(matcherSet.contains(oxygen));
        //
        assertTrue(matcherSet.add(hydrogen1));
        assertFalse(matcherSet.add(hydrogen1));
        assertTrue(matcherSet.contains(hydrogen1));
        assertFalse(matcherSet.contains(hydrogen2));
        //
        assertTrue(matcherSet.add(hydrogen2));
        assertFalse(matcherSet.add(hydrogen2));
        assertTrue(matcherSet.contains(hydrogen2));
        assertEquals(3, matcherSet.size());
    }
    
    @Test
    public void customMatcherSetTest() {
        Atom hydro = new Atom("Hydro", "H");
        Atom hydrogen = new Atom("Hydrogen", "H");
        Atom oxy = new Atom("Oxy", "O");
        Atom oxygen = new Atom("Oxygen", "O");
        
        Set<Atom> matcherSet = new MatcherLinkedHashSet<>(8, new Atom.SymbolMatcher());
        
        assertTrue(matcherSet.add(oxygen));
        assertFalse(matcherSet.add(oxy));
        assertFalse(matcherSet.add(oxygen));
        assertTrue(matcherSet.contains(oxy));
        assertTrue(matcherSet.contains(oxygen));
        //
        assertTrue(matcherSet.add(hydrogen));
        assertFalse(matcherSet.add(hydro));
        assertFalse(matcherSet.add(hydrogen));
        assertTrue(matcherSet.contains(hydro));
        assertTrue(matcherSet.contains(hydrogen));
        //
        assertEquals(2, matcherSet.size());
    }

}
