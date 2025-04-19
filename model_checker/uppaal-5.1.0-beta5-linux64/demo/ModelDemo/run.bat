@echo off

IF %UPPAAL_HOME%x == x (
    echo "Set UPPAAL_HOME environment variable to Uppaal installation path."
    pause
    exit 1
)

echo "Using UPPAAL_HOME=%UPPAAL_HOME%"
echo "=== COMPILING ==="
javac -cp "%UPPAAL_HOME%/lib/model.jar:%UPPAAL_HOME%/uppaal.jar" "ModelDemo.java"
IF %ERRORLEVEL% NEQ 0 (
    echo "Compilation failed with %ERROR_LEVEL%"
    pause
    exit %ERROR_LEVEL%
)
echo "=== RUNNING ==="
java -classpath ".:%UPPAAL_HOME%/lib/model.jar:%UPPAAL_HOME%/uppaal.jar" ModelDemo hardcoded
IF %ERRORLEVEL% NEQ 0 (
    echo "Compilation failed with %ERROR_LEVEL%"
    pause
    exit %ERROR_LEVEL%
)
