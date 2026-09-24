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
Link a JDK runtime image whose jdk.internal.vm.ci module already contains TornadoVM's frozen
JDK-21 JVMCI classes, so the launcher does not need --patch-module.

Why: on JDK 22-26 the launcher overlays share/java/jvmci/jvmci-21.0.2.jar onto the platform's
jdk.internal.vm.ci with --patch-module. HotSpot disables CDS whenever a module is patched, so
those command lines can neither use the JDK's default CDS archive nor create or map a Project
Leyden AOT cache (-XX:AOTCacheOutput / -XX:AOTCache, JDK 24+). Baking the overlay into the
image removes the flag; `tornado` detects the image through a marker in its `release` file.

How: rebuild jdk.internal.vm.ci.jmod with the overlay classes, re-record the module hashes that
java.base (and any other module) keeps for it, then jlink every module of the source JDK into
a new image and generate its default CDS archive.

The source JDK must ship a jmods/ directory. Some JDK 24+ builds omit it (JEP 493 run-time image
linking); jlink cannot substitute a system module in that mode, so use a distribution that
ships jmods (for example Amazon Corretto or Homebrew's openjdk).

Usage:
    tornado-jvmci-jdk.py --output <dir> [--java-home <jdk>] [--force]
    export JAVA_HOME=<dir>
    tornado --jvm="-XX:AOTCacheOutput=app.aot" -m <module>/<main>    # training run
    tornado --jvm="-XX:AOTCache=app.aot" -m <module>/<main>          # production runs
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

JVMCI_MODULE = "jdk.internal.vm.ci"
# Must match tornado.py, which skips --patch-module when the image's release file carries it.
RELEASE_MARKER = "TORNADOVM_JVMCI_OVERLAY"
# jmod extract directory -> jmod create option
JMOD_SECTIONS = (
    ("classes", "--class-path"),
    ("bin", "--cmds"),
    ("conf", "--config"),
    ("include", "--header-files"),
    ("legal", "--legal-notices"),
    ("lib", "--libs"),
    ("man", "--man-pages"),
)


def fail(msg):
    print("[ERROR] " + msg, file=sys.stderr)
    sys.exit(1)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        fail("command failed: " + " ".join(cmd) + "\n" + r.stdout + r.stderr)
    return r.stdout


def java_major(java_home):
    out = subprocess.run([os.path.join(java_home, "bin", "java"), "-version"], capture_output=True, text=True).stderr
    m = re.search(r'version\s+"(\d+)', out)
    if not m:
        fail("cannot determine the Java version of " + java_home)
    return int(m.group(1)), out


class Jmod:
    """What `jmod create` needs to faithfully recreate a jmod: version, platform and recorded hashes."""

    def __init__(self, jmod_tool, path):
        self.path = path
        text = run([jmod_tool, "describe", path])
        first = text.splitlines()[0].strip()
        self.name, _, self.version = first.partition("@")
        m = re.search(r"^platform (\S+)", text, re.M)
        self.platform = m.group(1) if m else None
        self.hashes = re.findall(r"^hashes (\S+) ", text, re.M)

    def create(self, jmod_tool, content_dir, out, module_path):
        cmd = [jmod_tool, "create"]
        for section, option in JMOD_SECTIONS:
            d = os.path.join(content_dir, section)
            if os.path.isdir(d):
                cmd += [option, d]
        if self.version:
            cmd += ["--module-version", self.version]
        if self.platform:
            cmd += ["--target-platform", self.platform]
        if self.hashes:
            cmd += ["--module-path", module_path, "--hash-modules", "^(" + "|".join(re.escape(h) for h in self.hashes) + ")$"]
        run(cmd + [out])


def main():
    parser = argparse.ArgumentParser(description="Link a JDK 22-26 image with TornadoVM's JVMCI overlay baked in (no --patch-module), "
                                                 "so it can use the default CDS archive and Project Leyden AOT caches.")
    parser.add_argument("--output", required=True, help="directory to create the JDK image in")
    parser.add_argument("--java-home", default=os.environ.get("JAVA_HOME"), help="source JDK (default: $JAVA_HOME)")
    parser.add_argument("--tornadovm-home", default=os.environ.get("TORNADOVM_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        help="TornadoVM SDK (default: $TORNADOVM_HOME)")
    parser.add_argument("--force", action="store_true", help="replace --output if it exists")
    args = parser.parse_args()

    if not args.java_home or not os.path.isdir(args.java_home):
        fail("set JAVA_HOME or pass --java-home")
    java_home = os.path.abspath(args.java_home)
    version, version_text = java_major(java_home)
    if version < 22 or version > 26:
        fail(f"JDK {version}: the launcher only patches {JVMCI_MODULE} on JDK 22-26, so there is nothing to bake in.")
    if version < 24:
        print(f"[WARNING] JDK {version} predates the AOT cache (JDK 24+); the image only gains the default CDS archive.")
    if "GraalVM" in version_text:
        fail("GraalVM JDKs are not supported by this tool.")

    jmods_dir = os.path.join(java_home, "jmods")
    if not os.path.isdir(jmods_dir) or not os.path.isfile(os.path.join(jmods_dir, JVMCI_MODULE + ".jmod")):
        fail(f"{java_home} has no jmods/ directory. This JDK was built for run-time image linking (JEP 493), where jlink "
             f"cannot replace {JVMCI_MODULE}. Use a JDK {version} distribution that ships jmods (e.g. Amazon Corretto, Homebrew openjdk).")

    overlays = sorted(glob.glob(os.path.join(args.tornadovm_home, "share", "java", "jvmci", "jvmci-*.jar")))
    if len(overlays) != 1:
        fail("expected exactly one share/java/jvmci/jvmci-*.jar in " + args.tornadovm_home + ", found " + str(overlays))
    overlay = overlays[0]
    overlay_name = os.path.splitext(os.path.basename(overlay))[0]

    output = os.path.abspath(args.output)
    if os.path.exists(output):
        if not args.force:
            fail(output + " exists (use --force to replace it)")
        shutil.rmtree(output)

    jmod_tool = os.path.join(java_home, "bin", "jmod")
    jlink_tool = os.path.join(java_home, "bin", "jlink")

    with tempfile.TemporaryDirectory(prefix="tornado-jvmci-jdk-") as tmp:
        mods = os.path.join(tmp, "jmods")
        shutil.copytree(jmods_dir, mods)
        modules = sorted(os.path.splitext(f)[0] for f in os.listdir(mods) if f.endswith(".jmod"))

        # 1. jdk.internal.vm.ci with the overlay classes. The platform's module-info.class is kept:
        #    the overlay's own descriptor is the JDK 21 one (the same one --patch-module ignores).
        ci_path = os.path.join(mods, JVMCI_MODULE + ".jmod")
        ci = Jmod(jmod_tool, ci_path)
        ci_dir = os.path.join(tmp, "ci")
        run([jmod_tool, "extract", "--dir", ci_dir, ci_path])
        classes = os.path.join(ci_dir, "classes")
        with zipfile.ZipFile(overlay) as z:
            for entry in z.namelist():
                if entry.endswith("/") or entry == "module-info.class" or entry.startswith("META-INF/"):
                    continue
                target = os.path.join(classes, entry)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(entry) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        os.remove(ci_path)
        ci.create(jmod_tool, ci_dir, ci_path, mods)

        # 2. Modules that record a hash of jdk.internal.vm.ci (java.base, through its qualified
        #    exports) would make jlink reject the modified module; recreate them with fresh hashes.
        for name in modules:
            if name == JVMCI_MODULE:
                continue
            path = os.path.join(mods, name + ".jmod")
            jm = Jmod(jmod_tool, path)
            if JVMCI_MODULE not in jm.hashes:
                continue
            content = os.path.join(tmp, "x-" + name)
            run([jmod_tool, "extract", "--dir", content, path])
            os.remove(path)
            jm.create(jmod_tool, content, path, mods)

        # 3. Link every module of the source JDK, so the image is a full JDK (javac, jcmd, ...).
        run([jlink_tool, "--module-path", mods, "--add-modules", ",".join(modules), "--generate-cds-archive", "--output", output])

    with open(os.path.join(output, "release"), "a") as f:
        f.write(f'{RELEASE_MARKER}="{overlay_name}"\n')

    print(f"Linked {output}")
    print(f"  JDK {version} with {JVMCI_MODULE} overlaid by {os.path.basename(overlay)}")
    print("Use it with:")
    print(f"  export JAVA_HOME={output}")
    print('  tornado --jvm="-XX:AOTCacheOutput=app.aot" -m <module>/<main-class>   # training run')
    print('  tornado --jvm="-XX:AOTCache=app.aot" -m <module>/<main-class>         # later runs')


if __name__ == "__main__":
    main()
