#!/usr/bin/env bash
set -e

# Find the location of this script
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DNAME=$(dirname "$SOURCE")
    HERE=$(cd -P "$DNAME" >/dev/null 2>&1 && pwd)
    SOURCE=$(readlink "$SOURCE")
    [[ "$SOURCE" != /* ]] && SOURCE="$HERE/$SOURCE"
done
DNAME=$(dirname "$SOURCE")
HERE=$(cd -P "$DNAME" > /dev/null 2>&1 && pwd)

if [ -z "$UPPAAL_HOME" ]; then
    echo "UPPAAL_HOME is not set, trying to find Uppaal installation"
    UPPAAL_HOME=$(realpath "$HERE")
    while [ ! "$UPPAAL_HOME" == "/" ]; do
        echo "Looking into $UPPAAL_HOME"
        if [ -r "$UPPAAL_HOME/uppaal.jar" ]; then
            break;
        else
            UPPAAL_HOME=$(realpath "$UPPAAL_HOME/..")
        fi
    done

    if [ ! -r "$UPPAAL_HOME/uppaal.jar" ] ; then
        echo "Could not find Uppaal installation."
        echo "Please set UPPAAL_HOME environment variable to Uppaal installation path."
        echo "For example on bash:"
        echo "export UPPAAL_HOME=/path/to/uppaal-5.1.0-linux64"
        exit 1
    fi
    export UPPAAL_HOME
fi

echo "=== COMPILING ==="
javac -cp "$UPPAAL_HOME/lib/model.jar:$UPPAAL_HOME/uppaal.jar" "$HERE/ModelDemo.java"
echo "=== RUNNING ==="
echo "java -classpath $UPPAAL_HOME/uppaal.jar:$HERE ModelDemo hardcoded"
exec java -classpath "$UPPAAL_HOME/uppaal.jar:$HERE" ModelDemo hardcoded
