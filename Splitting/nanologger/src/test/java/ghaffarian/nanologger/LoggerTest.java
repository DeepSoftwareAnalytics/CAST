/*** In The Name of Allah ***/
package ghaffarian.nanologger;

import static org.junit.Assert.*;
import org.junit.*;
import java.io.*;

/**
 * Unit tests for Nano-Logger.
 */
public class LoggerTest {
    
    private static final String LOG_NAME = "test.log";
    
    @BeforeClass
    public static void init() throws IOException {
        Logger.init(LOG_NAME);
    }
    
    @Before
    public void resetLogLevel() {
        Logger.setActiveLevel(Logger.Level.DEBUG); // All inclusive!
    }
    
    @Test
    public void testInit() {
        File test = new File(LOG_NAME);
        assertTrue(test.exists() && test.length() > 0);
    }
    
    @Test
    public void logTest() throws IOException {
        Logger.log("test1");
        Logger.log("test1 RAW", Logger.Level.RAW);
        Logger.log("test1 ERR", Logger.Level.ERROR);
        Logger.log("test1 WRN", Logger.Level.WARNING);
        Logger.log("test1 INF", Logger.Level.INFO);
        Logger.log("test1 DBG", Logger.Level.DEBUG);
        assertEquals(6, countOccurrenceOf("test1"));
    }
    
    @Test
    public void moreLogTest() throws IOException {
        Logger.log("test2");
        Logger.error("test2 ERR");
        Logger.warn("test2 WRN");
        Logger.info("test2 INF");
        Logger.debug("test2 DBG");
        assertEquals(5, countOccurrenceOf("test2"));
    }
    
    @Test
    public void activeLevelTest() throws IOException {
        Logger.setActiveLevel(Logger.Level.WARNING);
        Logger.log("test3");
        Logger.error("test3 ERR");
        Logger.warn("test3 WRN");
        Logger.info("test3 INF");
        Logger.debug("test3 DBG");
        assertEquals(3, countOccurrenceOf("test3"));
        //
        Logger.setActiveLevel(Logger.Level.ERROR);
        Logger.log("test4");
        Logger.error("test4 ERR");
        Logger.warn("test4 WRN");
        Logger.info("test4 INF");
        Logger.debug("test4 DBG");
        assertEquals(2, countOccurrenceOf("test4"));
        //
        Logger.setActiveLevel(Logger.Level.INFO);
        Logger.log("test5");
        Logger.error("test5 ERR");
        Logger.warn("test5 WRN");
        Logger.info("test5 INF");
        Logger.debug("test5 DBG");
        assertEquals(4, countOccurrenceOf("test5"));
    }
    
    @Test
    public void timeDateTest() throws IOException {
        Logger.log("Current time is: " + Logger.time());
        Logger.log("Current date is: " + Logger.date());
        assertEquals(2, countOccurrenceOf("Current "));
    }
    
    @Test
    public void exceptionTest() throws IOException {
        try {
            throwTestException();
        } catch(RuntimeException ex) {
            Logger.log(ex, Logger.Level.INFO);
            Logger.warn(ex);
            Logger.error(ex);
        }
        assertEquals(6, countOccurrenceOf("test exception"));
    }
    
    @Test
    public void printfTest() throws IOException {
        Logger.printf("printf no args");
        Logger.printf("printf %d %s 0x%X", 1234, "5678", 3735928559L);
        Logger.printf(Logger.Level.WARNING, "printf warning no args");
        Logger.printf(Logger.Level.INFO, "printf info %d %s 0x%X", 1234, "5678", 3735928559L);
        assertEquals(4, countOccurrenceOf("printf"));
    }
    
    @Test
    public void stdErrorRedirectTest() throws IOException {
        Logger.redirectStandardError("test");
        System.err.println("std error test 1");
        System.err.println("std error test 2");
        System.err.println("std error test 3");
        int counter = 0;
        try (BufferedReader reader = new BufferedReader(new FileReader("test.err"))) {
            while (reader.ready()) {
                String line = reader.readLine();
                if (line.contains("std error test")) {
                    counter++;
                }
            }
        }
        assertEquals(3, counter);
    }
    
    @AfterClass
    public static void close() {
        Logger.close();
    }

    
    private void throwTestException() {
        throw new RuntimeException("This is just a test exception");
    }
    
    private int countOccurrenceOf(String pattern) throws IOException {
        int counter = 0;
        try (BufferedReader reader = new BufferedReader(new FileReader(LOG_NAME))) {
            while (reader.ready()) {
                String line = reader.readLine();
                if (line.contains(pattern)) {
                    counter++;
                }
            }
        }
        return counter;
    }
}
