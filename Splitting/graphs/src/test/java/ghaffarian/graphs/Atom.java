/*** In The Name of Allah ***/
package ghaffarian.graphs;

import java.util.Objects;

/**
 * Model of an atom.
 *
 * @author Seyed Mohammad Ghaffarian
 */
public class Atom {

    public final String NAME;
    public final String SYMB;

    public Atom(String name, String symbol) {
        this.NAME = name;
        this.SYMB = symbol;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        final Atom other = (Atom) obj;
        return Objects.equals(this.SYMB, other.SYMB) && Objects.equals(this.NAME, other.NAME);
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 37 * hash + Objects.hashCode(this.NAME);
        hash = 37 * hash + Objects.hashCode(this.SYMB);
        return hash;
    }
    
    static class SymbolMatcher implements Matcher<Atom> {

        @Override
        public boolean equals(Atom o1, Atom o2) {
            if (o1 == null || o2 == null)
                return false;
            if (o1 == o2)
                return true;
            return (o1.SYMB).equals(o2.SYMB);
        }

        @Override
        public int hashCode(Atom o) {
            return o.SYMB.hashCode();
        }

    }
}
