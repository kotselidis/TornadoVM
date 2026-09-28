#!/usr/bin/env bash
# Builds the M0 spike module and runs it with TornadoVM's Metal backend.
# Usage: [MAIN=DecodeOps] prototypes/mlx-m0/java/run.sh [extra JVM flags, e.g. -Dtornado.print.bytecodes=True]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../../../setvars.sh"

OUT="$HERE/build"
rm -rf "$OUT" && mkdir -p "$OUT/classes" "$OUT/mods"

"$JAVA_HOME/bin/javac" --source 21 --enable-preview -Xlint:-preview \
    --module-path "$TORNADOVM_HOME/share/java/tornado" \
    --upgrade-module-path "$TORNADOVM_HOME/share/java/graalJars" \
    -d "$OUT/classes" $(find "$HERE/src" -name '*.java')
"$JAVA_HOME/bin/jar" --create --file "$OUT/mods/tornado-mlx-spike.jar" -C "$OUT/classes" .

# The generated argfile puts "." first on the module path, so run from the jar's directory.
cd "$OUT/mods"
exec "$JAVA_HOME/bin/java" @"$TORNADOVM_HOME/tornado-argfile" --enable-native-access=tornado.mlx.spike \
    --add-opens tornado.drivers.metal/uk.ac.manchester.tornado.drivers.metal=tornado.mlx.spike "$@" \
    -m tornado.mlx.spike/uk.ac.manchester.tornado.mlx.spike.${MAIN:-SpikeMain}
