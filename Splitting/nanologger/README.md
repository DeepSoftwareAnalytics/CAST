# Nano-Logger
*A Simple, Nimble, Thread-safe Logging Utility in Java*. 

While there are many great logging utilities for Java ( such as [Log4j](https://logging.apache.org/log4j/), [Logback](https://logback.qos.ch/), [Java Core Logging API](https://docs.oracle.com/javase/8/docs/api/java/util/logging/package-summary.html) ) sometimes you just need something plain simple, without adding any additional dependencies, XML configurations, DTDs, etc. That's exactly why I developed my own simple logging utility in Java: **Nano-Logger**.

 - It's ***simple***, because it's all a single class with simple static methods (no need for instantiation, object sharing, complex initialization and configurations), consisting an easy API with clear javadoc!
 - It's ***nimble***, because it's all a single class, using only standard Java, without any 3rd-party dependencies, no XML configurations, and no DTDs!
 - It's ***thread-safe***, because all static logging operations are atomic, and can be safely called from different threads, without worrying about synchronization!

## How to use it?
This simple utility can be used in two ways:
 1. Either copy the single `Logger` class to a proper package in your project source-code.
 2. Or add the JAR release as a 3rd-party library to your project CLASSPATH.

## What about licensing issues?
Nano-Logger is licensed under the terms of the **Apache-v2.0**. If you read the LICENSE, you'll see there's no problem using it in your open-source or proprietary software project.

## What does the API look like?
Just check out the public interface of the single `Logger` class and it's javadoc. It's just plain simple and easy to use!

You can also check out the JUnit test codes, which serve as a live and executable API documentation.

Here's a very basic example usage:

    // Initialize the logger with a desired path for your log file
    Logger.init("/path/to/log/file.log");

    // Now start logging!
    Logger.log("some log message!");
    Logger.log("another log message!");

The above code shows raw logging (without any time-tags and labels). It's exactly like printing string-messages in a text file. The `init` method only needs to be called once for any logging session.

Another more advanced logging scenario (more common in larger software projects) is logging with time-tags and various log-level labels:

    // set the active log-level
    Logger.setActiveLevel(Logger.Level.WARNING);

    // now do some logging at various levels
    Logger.log("some warning message", Logger.Level.WARNING);
    Logger.log("some error message", Logger.Level.ERROR);
    Logger.log("some debug message", Logger.Level.DEBUG);

The result of the above three log operations will look like below:

    Thu 2018/Oct/04 10:52:18:923 [WRN] | some warning message
    Thu 2018/Oct/04 10:52:18:923 [ERR] | some error message


Note that the debug message is missing. That's because we have set the active log level to `WARNING`; so any log operations above the active-level will be ignored and not written to the log file.

There are also short-hand methods for the various log-levels available:

    // short-hand log operations in ascending order
    Logger.log("some raw message");
    Logger.printf("%s", "another raw message!");
    Logger.error("some error message");
    Logger.warn("some warning message");
    Logger.info("some information message");
    Logger.debug("some debug message");


## Where is this project going?
Although I'm already satisfied with the current state of this utility class, I'm considering to make the API consistent with one of the well-known logging-API facades (such as [SLF4J](https://www.slf4j.org/) or [Apache Commons Logging](http://commons.apache.org/proper/commons-logging/)). I haven't thoroughly studied this yet; so, this will only happen if it doesn't conflict with the main goals of the project (***being simple and nimble***).

Other than this, if you have any bright ideas or feature request, you are highly welcome to open an issue, or even contribute to this project. Of course, contributions or feature-requests which do not conflict with the main goals are acceptable. Remember, I don't intend to create another bloated/complicated logging library in Java!
