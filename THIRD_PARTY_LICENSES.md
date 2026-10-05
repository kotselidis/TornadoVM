# Third-party components

The TornadoVM SDK ships the following third-party components in `share/java`. Each keeps its own
license; the license texts are in the SDK root or inside the component's jar (`META-INF`).

## Graal compiler and GraalVM SDK

| Component | Version | Shipped as | License |
|---|---|---|---|
| Graal compiler (`org.graalvm.compiler:compiler`), relocated to `tornado.graal` | 23.1.0 | `tornado-graal-23.1.0.jar` | GPLv2 with Classpath Exception ([LICENSE_GPLv2CE](LICENSE_GPLv2CE)), Oracle and/or its affiliates |
| GraalVM word (`org.graalvm.sdk:word`) | 23.1.0 | `word-23.1.0.jar` | Universal Permissive License 1.0 ([LICENSE_UPL](LICENSE_UPL)) |
| GraalVM collections (`org.graalvm.sdk:collections`) | 23.1.0 | `collections-23.1.0.jar` | Universal Permissive License 1.0 ([LICENSE_UPL](LICENSE_UPL)) |
| Truffle compiler API (`org.graalvm.truffle:truffle-compiler`) | 23.1.0 | `truffle-compiler-23.1.0.jar` | Universal Permissive License 1.0 ([LICENSE_UPL](LICENSE_UPL)) |
| JVMCI API from OpenJDK (`jdk.internal.vm.ci`), repackaged | 21.0.2 | `jvmci-21.0.2.jar` | GPLv2 ([LICENSE_GPLv2](LICENSE_GPLv2)), Oracle and/or its affiliates |

## Libraries

| Component | Version | Used by | License |
|---|---|---|---|
| ASM | 9.7 | tornado-annotation, tornado-runtime | BSD-3-Clause |
| SNMP4J | 2.8.6 | tornado-runtime (UPS power metering) | Apache 2.0 ([LICENSE_APACHE2](LICENSE_APACHE2)) |
| Apache Log4j API and Core | 2.25 | tornado-runtime | Apache 2.0 |
| EJML (core, ddense, dsparse, simple) | 0.38 | tornado-matrices | Apache 2.0 |
| Apache Lucene core | 8.2.0 | tornado-unittests | Apache 2.0 |
| JUnit | 4.13.2 | tornado-unittests | Eclipse Public License 1.0 |
| Hamcrest core | 1.3 | tornado-unittests (via JUnit) | BSD-3-Clause |
| JMH core | 1.29 | tornado-benchmarks | GPLv2 with Classpath Exception |
| JOpt Simple | 4.6 | tornado-benchmarks (via JMH) | MIT |
| Apache Commons Math | 3.6.1 | tornado-benchmarks (via JMH) | Apache 2.0 |
| Apache Commons Lang | 3.18.0 | tornado-benchmarks | Apache 2.0 |
| JSR-305 annotations | 3.0.2 | transitive | Apache 2.0 |
