#  AST_Splitting_Tools


AST_Splitting_Tools is revise from [PROGEX](https://github.com/ghaffarian/progex) :Program Graph Extractor which can extract well-known graphical program representations (such as CFGs, PDGs, ASTs, etc.) 
from software source code.

asts (AST Splitting Tools) can exact AST and split it into subtrees in a hierarchically way.
from source code and export it into dot  file.

## Usage Guide



```
USAGE:

   java -jar asts.jar [-OPTIONS...] /path/to/program/src


OPTIONS:

   -help      Print this help message
   -outdir    Specify path of output directory
   -format    Specify output format; either 'DOT', 'GML' or 'JSON'
   -lang      Specify language of program source codes
   -node_level     Specify  note level:  block (just support AST)
   
   -ast       Perform AST (Abstract Syntax Tree) analysis
   -cfg       Perfomt CFG (Control Flow Graph) analysis
   -icfg      Perform ICFG (Interprocedural CFG) analysis
   -info      Analyze and extract detailed information about program source code
   -pdg       Perform PDG (Program Dependence Graph) analysis

   -debug     Enable more detailed logs (only for debugging)
   -timetags  Enable time-tags and labels for logs (only for debugging)


DEFAULTS:

   - If not specified, the default output directory is the current working directory.
   - If not specified, the default output format is DOT.
   - If not specified, the default language is Java.
   - There is no default value for analysis type.
   - There is no default value for input directory path.

```
### Run asts.jar 

    java -jar asts.jar -outdir ./example/dot  -ast -lang java -format dot  -node_level block  ./example/java
This example will extract the Splitted AST of all Java source files in the given path (`./example/java`) and 
will export all extracted graphs as dot files in the given output directory.
    

