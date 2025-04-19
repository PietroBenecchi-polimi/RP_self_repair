# UPPAAL Java API Demo

This directory contains a sample Java project code demonstrating the use of Java API
to manipulate Uppaal files (models and traces).

The source code assumes that this project is inside Uppaal distribution and can find Uppaal files.
Alternatively set UPPAAL_HOME environment variable to Uppaal installation path.

The `lib/model.jar` library offers an API to create Uppaal model documents.
The `lib/model-javadoc.jar` includes the Javadoc documentation for this library.
In particular see `com.uppaal.model.core2.Document` class documentation for creating model documents.

The library can also be used to connect to engine (for simulation and verification), but then `uppaal.jar` must be included on class path.
See the documentation for `com.uppaal.engine.Engine` class to connect to Uppaal engine.

Use `run.sh` on Linux/macOS and `run.bat` on Windows to compile and run.

The program will produce `ModelDemo.xml` model file, a some sample traces `ModelDemo.xtr` and
some human friendly messages to standard output.

`ModelDemo` can also read an external model file (use Control+C to stop simulation):

```sh
java -cp uppaal.jar:demo/ModelDemo ModelDemo demo/train-gate.xml
```

Marius Mikucionis
marius@cs.aau.dk
