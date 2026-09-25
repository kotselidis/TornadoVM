#!/usr/bin/env python3
#
# Copyright (c) 2026, APT Group, Department of Computer Science,
# The University of Manchester.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Generates the FFM bindings for mlx-c (the C API of Apple MLX) used by tornado-mlx.

Reads the mlx-c headers and writes:
  * src/main/java/uk/ac/manchester/tornado/mlx/provider/MlxC.java
      one static method per mlx-c function, backed by a lazily resolved downcall;
  * mlx-c-api.json
      the operation list (header, name, parameters) that update_coverage.py turns
      into the coverage manifest.

Both outputs are checked in, so building tornado-mlx does not need the headers.
Re-run after upgrading mlx-c:

    python3 tornado-mlx/scripts/generate_bindings.py [--include /opt/homebrew/opt/mlx-c/include]

The generator only understands the C types listed in this file and stops on
anything else, so an mlx-c upgrade that introduces a new type fails loudly.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAVA_OUT = os.path.join(MODULE_DIR, "src/main/java/uk/ac/manchester/tornado/mlx/provider/MlxC.java")
API_OUT = os.path.join(MODULE_DIR, "mlx-c-api.json")

# Headers whose functions are MLX operations: these are tracked in the coverage manifest.
OP_HEADERS = ["ops.h", "fast.h", "linalg.h", "fft.h", "random.h", "transforms.h"]

# Support headers: array creation and access, streams, devices, vectors, strings, memory.
# Bound, but not operations, so not part of the coverage manifest.
SUPPORT_HEADERS = ["array.h", "stream.h", "device.h", "vector.h", "string.h", "memory.h", "metal.h", "version.h", "error.h"]

# Functions in the headers above that are not bound, with the reason. They take C callbacks
# (function pointers) that need upcall plumbing of their own.
SKIPPED = {
    "mlx_set_error_handler": "takes a C callback; errors are reported through return codes instead",
}

JAVA_KEYWORDS = {"abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue", "default", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long", "native", "new", "package",
                 "private", "protected", "public", "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try",
                 "void", "volatile", "var", "record", "yield"}

# Scalar C types -> (Java type, FFM layout expression)
SCALARS = {
    "int": ("int", "C_INT"),
    "unsigned int": ("int", "C_INT"),
    "uint32_t": ("int", "C_INT"),
    "int32_t": ("int", "C_INT"),
    "size_t": ("long", "C_LONG"),
    "uint64_t": ("long", "C_LONG"),
    "int64_t": ("long", "C_LONG"),
    "bool": ("boolean", "C_BOOL"),
    "float": ("float", "C_FLOAT"),
    "double": ("double", "C_DOUBLE"),
}

# Structs passed by value that are not one-pointer handles.
STRUCTS = {
    "mlx_optional_int": "OPT_INT",
    "mlx_optional_float": "OPT_FLOAT",
    "mlx_optional_dtype": "OPT_DTYPE",
}


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def discover_types(include_dir):
    """One-pointer handle structs and enums declared anywhere in mlx-c."""
    handles, enums = set(), set()
    for name in sorted(os.listdir(include_dir)):
        if not name.endswith(".h"):
            continue
        text = strip_comments(open(os.path.join(include_dir, name)).read())
        for m in re.finditer(r"typedef\s+struct\s+\w+\s*\{([^}]*)\}\s*(\w+)\s*;", text):
            body = " ".join(m.group(1).split())
            if body == "void* ctx;":
                handles.add(m.group(2))
        for m in re.finditer(r"typedef\s+enum\s+\w*\s*\{[^}]*\}\s*(\w+)\s*;", text):
            enums.add(m.group(1))
    return handles, enums


def parse_prototypes(path):
    text = strip_comments(open(path).read())
    text = re.sub(r"#[^\n]*", " ", text)
    protos = []
    for m in re.finditer(r"([A-Za-z_][\w\s\*]*?)\b(mlx_\w+)\s*\(([^;{}]*)\)\s*;", text):
        ret = " ".join(m.group(1).split())
        if ret.startswith("typedef") or "(" in ret:
            continue
        params = " ".join(m.group(3).split())
        protos.append((ret, m.group(2), params))
    return protos


def split_param(param, index):
    """'const mlx_array a' -> ('mlx_array', False, 'a'); 'const int* shape' -> ('int', True, 'shape')."""
    param = param.strip()
    m = re.match(r"^(.*?)([A-Za-z_]\w*)$", param)
    ctype, name = (m.group(1), m.group(2)) if m else (param, "p%d" % index)
    if ctype.strip() in ("", "const", "struct"):
        # Unnamed parameter: the whole thing is the type.
        ctype, name = param, "p%d" % index
    pointer = "*" in ctype
    base = " ".join(ctype.replace("*", " ").replace("const", " ").split())
    return base, pointer, name


class TypeMapper:

    def __init__(self, handles, enums):
        self.handles = handles
        self.enums = enums

    def map(self, base, pointer, where):
        if pointer:
            return "MemorySegment", "C_POINTER", "pointer"
        if base in SCALARS:
            java, layout = SCALARS[base]
            return java, layout, "scalar"
        if base in self.enums:
            return "int", "C_INT", "enum"
        if base in self.handles:
            return "MemorySegment", "C_POINTER", "handle"
        if base in STRUCTS:
            return "MemorySegment", STRUCTS[base], "struct"
        raise SystemExit("[generate_bindings] unsupported C type '%s' in %s" % (base, where))


def java_name(name):
    """C parameter name -> checkstyle-clean Java name: lower camelCase, not a keyword."""
    parts = [p for p in name.split("_") if p]
    camel = parts[0].lower() + "".join(p[:1].upper() + p[1:].lower() for p in parts[1:]) if parts else name
    return camel + "Value" if camel in JAVA_KEYWORDS else camel


MAX_LINE = 180


def wrap_items(prefix, items, suffix, indent):
    """prefix + ', '.join(items) + suffix, broken across lines of at most MAX_LINE characters."""
    one = prefix + ", ".join(items) + suffix
    if len(one) <= MAX_LINE:
        return [one]
    lines, current = [], prefix
    for i, item in enumerate(items):
        piece = item + (", " if i < len(items) - 1 else suffix)
        if len(current) + len(piece) > MAX_LINE and current.strip():
            lines.append(current.rstrip())
            current = indent
        current += piece
    lines.append(current.rstrip())
    return lines


def mlx_c_version(include_dir):
    real = os.path.realpath(include_dir)
    m = re.search(r"/mlx-c/([0-9][^/]*)/", real + "/")
    if m:
        return m.group(1)
    try:
        out = subprocess.run(["brew", "list", "--versions", "mlx-c"], capture_output=True, text=True, check=False).stdout.split()
        return out[1] if len(out) > 1 else "unknown"
    except OSError:
        return "unknown"


def default_include():
    try:
        prefix = subprocess.run(["brew", "--prefix", "mlx-c"], capture_output=True, text=True, check=True).stdout.strip()
        return os.path.join(prefix, "include")
    except (OSError, subprocess.CalledProcessError):
        return "/opt/homebrew/opt/mlx-c/include"


def generate(include_dir):
    c_dir = os.path.join(include_dir, "mlx", "c")
    if not os.path.isdir(c_dir):
        raise SystemExit("[generate_bindings] no mlx-c headers under %s" % c_dir)
    handles, enums = discover_types(c_dir)
    mapper = TypeMapper(handles, enums)
    version = mlx_c_version(include_dir)

    functions = []  # (header, is_op, name, ret(java, layout), [(java, layout, kind, cname, ctype_text)])
    digest = hashlib.sha256()
    for header in OP_HEADERS + SUPPORT_HEADERS:
        path = os.path.join(c_dir, header)
        if not os.path.exists(path):
            raise SystemExit("[generate_bindings] missing header %s" % path)
        digest.update(open(path, "rb").read())
        for ret, name, params in parse_prototypes(path):
            if name in SKIPPED:
                continue
            where = "%s:%s" % (header, name)
            if ret == "void":
                ret_map = None
            else:
                rbase = " ".join(ret.replace("*", " ").replace("const", " ").split())
                ret_map = mapper.map(rbase, "*" in ret, where + " (return)")
            plist = []
            if params and params != "void":
                for i, p in enumerate(params.split(",")):
                    base, pointer, pname = split_param(p, i)
                    java, layout, kind = mapper.map(base, pointer, where)
                    plist.append((java, layout, kind, java_name(pname), p.strip()))
            functions.append((header, header in OP_HEADERS, name, ret_map, plist))

    names = [f[2] for f in functions]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise SystemExit("[generate_bindings] duplicate functions: %s" % ", ".join(sorted(dupes)))
    return functions, version, digest.hexdigest()[:16]


def write_java(functions, version, digest):
    out = []
    w = out.append
    w("""/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
// GENERATED by tornado-mlx/scripts/generate_bindings.py from mlx-c %(version)s (headers %(digest)s). Do not edit.
package uk.ac.manchester.tornado.mlx.provider;

import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_DOUBLE;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_FLOAT;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_INT;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_LONG;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_POINTER;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;

/**
 * Raw FFM bindings to every function of the mlx-c operation headers (%(ops)s)
 * and support headers (%(support)s),
 * one static method per C function with the C name. Downcalls are resolved on first use, so a host without libmlxc only
 * fails when an MLX function is actually called.
 *
 * <p>
 * mlx-c handles ({@code mlx_array}, {@code mlx_stream}, {@code mlx_vector_array}, ...) are
 * {@code struct { void* ctx; }} passed by value, which the LP64 ABIs MLX supports pass and return
 * exactly like a pointer, so they appear here as {@link MemorySegment} addresses. Out-parameters
 * ({@code mlx_array* res}) are pointers to such a handle. {@code mlx_optional_*} structs are passed
 * by value with the layouts below. Functions return the mlx-c status code (0 on success) unless
 * noted otherwise.
 * </p>
 */
public final class MlxC {

    /** mlx-c version these bindings were generated from. */
    public static final String MLX_C_VERSION = "%(version)s";

    static final ValueLayout.OfBoolean C_BOOL = ValueLayout.JAVA_BOOLEAN;

    /** {@code mlx_optional_int}: {@code { int value; bool has_value; }}. */
    public static final StructLayout OPT_INT = MemoryLayout.structLayout(C_INT.withName("value"), C_BOOL.withName("has_value"), MemoryLayout.paddingLayout(3));

    /** {@code mlx_optional_float}: {@code { float value; bool has_value; }}. */
    public static final StructLayout OPT_FLOAT = MemoryLayout.structLayout(C_FLOAT.withName("value"), C_BOOL.withName("has_value"), MemoryLayout.paddingLayout(3));

    /** {@code mlx_optional_dtype}: {@code { mlx_dtype value; bool has_value; }}. */
    public static final StructLayout OPT_DTYPE = MemoryLayout.structLayout(C_INT.withName("value"), C_BOOL.withName("has_value"), MemoryLayout.paddingLayout(3));

    private static final SymbolLookup LIB = FFMSupport.loadLibrary("libmlxc.dylib", "/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib", "/usr/local/opt/mlx-c/lib/libmlxc.dylib",
            "libmlxc.so");
""" % {"version": version, "digest": digest, "ops": ", ".join(OP_HEADERS), "support": ", ".join(SUPPORT_HEADERS)})

    w("    private static final String[] NAMES = {")
    for i, f in enumerate(functions):
        w('            "%s",' % f[2])
    w("    };\n")
    w("    private static final FunctionDescriptor[] DESCRIPTORS = {")
    for f in functions:
        ret, params = f[3], f[4]
        layouts = [p[1] for p in params]
        if ret is None:
            out.extend(wrap_items("            FunctionDescriptor.ofVoid(", layouts, "),", " " * 20))
        else:
            out.extend(wrap_items("            FunctionDescriptor.of(", [ret[1]] + layouts, "),", " " * 20))
    w("    };\n")
    w("""    private static final MethodHandle[] HANDLES = new MethodHandle[NAMES.length];

    private MlxC() {
    }

    /** Whether libmlxc could be loaded on this host. */
    public static boolean isLoaded() {
        return LIB != null;
    }

    private static MethodHandle handle(int id) {
        MethodHandle h = HANDLES[id];
        if (h == null) {
            if (LIB == null) {
                throw new TornadoRuntimeException("[ERROR] Unable to load mlx-c. Install it (e.g. `brew install mlx-c`) and make sure libmlxc is on the library path.");
            }
            h = FFMSupport.downcall(LIB, DESCRIPTORS[id], NAMES[id]);
            if (h == null) {
                throw new TornadoRuntimeException("[ERROR] mlx-c symbol not found: " + NAMES[id]);
            }
            HANDLES[id] = h;
        }
        return h;
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) {
            return r;
        }
        if (t instanceof Error e) {
            throw e;
        }
        return new TornadoRuntimeException((Exception) t);
    }
""")
    current = None
    for i, f in enumerate(functions):
        header, _, name, ret, params = f
        if header != current:
            w("    // ---------------------------------------------------------------- %s\n" % header)
            current = header
        sig = ["%s %s" % (p[0], p[3]) for p in params]
        args = [p[3] for p in params]
        c_params = [p[4].replace("*/", "* /") for p in params] or ["void"]
        jret = "void" if ret is None else ret[0]
        w("    /**")
        w("     * Calls {@code %s}." % name)
        w("     *")
        out.extend(wrap_items("     * <p>C: {@code %s(" % name, c_params, ")}", "     *     "))
        w("     */")
        out.extend(wrap_items("    public static %s %s(" % (jret, name), sig, ") {", " " * 12))
        w("        try {")
        if ret is None:
            out.extend(wrap_items("            handle(%d).invokeExact(" % i, args, ");", " " * 20))
        else:
            out.extend(wrap_items("            return (%s) handle(%d).invokeExact(" % (jret, i), args, ");", " " * 20))
        w("        } catch (Throwable t) {")
        w("            throw rethrow(t);")
        w("        }")
        w("    }\n")
    w("}")
    with open(JAVA_OUT, "w") as fh:
        fh.write("\n".join(out) + "\n")


def write_api(functions, version, digest):
    ops = []
    for header, is_op, name, ret, params in functions:
        if not is_op:
            continue
        ops.append({
            "name": name,
            "header": header,
            "params": [p[4] for p in params],
        })
    data = {
        "generatedBy": "tornado-mlx/scripts/generate_bindings.py",
        "mlxCVersion": version,
        "headersDigest": digest,
        "operations": ops,
    }
    with open(API_OUT, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    return len(ops)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include", default=default_include(), help="mlx-c include directory (contains mlx/c/*.h)")
    args = parser.parse_args()
    functions, version, digest = generate(args.include)
    write_java(functions, version, digest)
    n_ops = write_api(functions, version, digest)
    print("[generate_bindings] mlx-c %s: %d functions bound (%d operations), %d skipped -> %s, %s" % (version, len(functions), n_ops, len(SKIPPED), os.path.relpath(JAVA_OUT),
                                                                                                          os.path.relpath(API_OUT)))


if __name__ == "__main__":
    sys.exit(main())
