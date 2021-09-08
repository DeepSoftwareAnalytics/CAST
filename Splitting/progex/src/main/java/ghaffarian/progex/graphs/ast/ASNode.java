/*** In The Name of Allah ***/
package ghaffarian.progex.graphs.ast;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Class type of Abstract Syntax (AS) nodes.
 * 
 * @author Seyed Mohammad Ghaffarian
 */
public class ASNode {
	
    /**
     * Enumeration of different types for AS nodes.
     */
    public enum Type {
        ROOT        ("ROOT"),
        IMPORTS     ("IMPORTS"),
        IMPORT      ("IMPORT"),
        PACKAGE     ("PACKAGE"),
        NAME        ("NAME"),
        MODIFIER    ("MODIFIER"),
        CLASS       ("CLASS"),
        EXTENDS     ("EXTENDS"),
        IMPLEMENTS  ("IMPLEMENTS"),
        INTERFACE   ("INTERFACE"),
        STATIC_BLOCK("STATIC-BLOCK"),
        CONSTRUCTOR ("CONSTRUCTOR"),
        FIELD       ("FIELD"),
        TYPE        ("TYPE"),
        METHOD      ("METHOD"),
        METHOD_SIGN ("METHOD_SIGNATURWE"),
        METHOD_BODY ("METHOD_BODY"),
        RETURN      ("RETURN"),
        PARAMS      ("PARAMS"),
        BLOCK       ("BLOCK"),
        IF          ("IF"),
        NESTED_IF   ("NESTED_IF"),
        CONDITION   ("COND"),
        THEN        ("THEN"),
        ELSE        ("ELSE"),
        VARIABLE    ("VAR"),
        INIT_VALUE  ("INIT"),
        STATEMENT   (""),
        STMTS       ("STATEMENTS_BLOCK"),
        FOR         ("FOR"),
        NESTED_FOR  ("NESTED_FOR"),
        FOR_INIT    ("INIT"),
        FOR_UPDATE  ("UPDATE"),
        FOR_EACH    ("FOR-EACH"),
        IN          ("IN"),
        WHILE       ("WHILE"),
        NESTED_WHILE("NESTED_WHILE"),
        DO_WHILE    ("DO-WHILE"),
        NESTED_DO_WHILE ("NESTED_DO-WHILE"),
        TRY         ("TRY"),
        NESTED_TRY  ("NESTED_TRY"),
        RESOURCES   ("RESOURCES"),
        CATCH       ("CATCH"),
        FINALLY     ("FINALLY"),
        SWITCH      ("SWITCH"),
        NESTED_SWITCH ("NESTED_SWITCH"),
        CASE        ("CASE"),
        DEFAULT     ("DEFAULT"),
        LABELED     ("LABELED"),
        NESTED_LABELED("NESTED_LABELED"),
        SYNC        ("SYNCHRONIZED"),
    	NESTED_SYNC ("NESTED_SYNCHRONIZED");

        public final String label;

        private Type(String lbl) {
            label = lbl;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    
    private Map<String, Object> properties;

    public ASNode(Type type) {
        properties = new LinkedHashMap<>();
        setLineOfCode(0);
        setType(type);
    }

    public final void setType(Type type) {
        properties.put("type", type);
    }

    public final Type getType() {
        return (Type) properties.get("type");
    }

    public final void setLineOfCode(int line) {
        properties.put("line", line);
    }

    public final int getLineOfCode() {
        return (Integer) properties.get("line");
    }

    public final void setCode(String code) {
        properties.put("code", code);
    }

    public final String getCode() {
        return (String) properties.get("code");
    }
    
    public final void setNormalizedCode(String normal) {
        if (normal != null)
            properties.put("normalized", normal);
    }

    public final String getNormalizedCode() {
        String normalized = (String) properties.get("normalized");
        if (normalized != null && !normalized.isEmpty())
            return normalized;
        return (String) properties.get("code");
    }
    
    public final void setProperty(String key, Object value) {
        properties.put(key.toLowerCase(), value);
    }

    public Object getProperty(String key) {
        return properties.get(key.toLowerCase());
    }

    public Set<String> getAllProperties() {
        return properties.keySet();
    }

    @Override
    public String toString() {
    	
//    	return getNormalizedCode();
        String code = getCode();
        if (code == null || code.isEmpty())
            return getType().label;
        if (getType().label.isEmpty())
//            return getLineOfCode() + ":  " + code;
        	return  code;
        return getType().label + ": " + code;
    }
}
