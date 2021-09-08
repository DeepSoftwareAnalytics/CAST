# Graphs: a Graph Library in Java
Graphs is a graph library in Java (such as [JGraphT](http://jgrapht.org)), designed and implemented by SM Ghaffarian.

## FAQ
#### 1. Why Create Another Graph Library?
Although [JGraphT](http://jgrapht.org) is a fine library, its model of graphs is too simplistic and limited, and on the other hand, its implementation is too complicated! These design flaws make it unsuitable for some complex graph applications (such as graph-mining).

For example, consider chemical molecules (e.g. water: H2O) which might have several instances of the same atom (2x Hydrogen vertices) and similar edges (2x H--O covalent bonds).

A JGraphT graph object does not allow duplicate vertices, nor duplicate edges. This is consistent with the mathematical definition of graphs; but in our experience, the JGrpahT API makes it difficult to handle such scenarios in an elegant way.

#### 2. Why not contribute to JGraphT?

I did consider this, but the class hierarchy of JGraphT and its implementation was too complex and I really didn't like the complex design of the classes. Also, some basic design choices had to be changed (like the model of an Edge) which simply wasn't compatible with the current design of JGraphT.
