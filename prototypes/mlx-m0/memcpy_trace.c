/*
 * Interposes memcpy/memmove and logs every copy of at least MEMCPY_TRACE_MIN bytes
 * (default 64 KiB). Used with zerocopy_probe to show that wrapping a TornadoVM-style
 * MTLBuffer as an MLX array does not copy it (stands in for an Instruments trace).
 *
 *   DYLD_INSERT_LIBRARIES=build/libmemcpy_trace.dylib build/zerocopy_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DYLD_INTERPOSE(_replacement, _replacee)                                                                                           \
    __attribute__((used)) static struct {                                                                                                 \
        const void *replacement;                                                                                                          \
        const void *replacee;                                                                                                             \
    } _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = { (const void *) (unsigned long) &_replacement,              \
        (const void *) (unsigned long) &_replacee };

static size_t minBytes(void) {
    static size_t cached = 0;
    if (cached == 0) {
        const char *env = getenv("MEMCPY_TRACE_MIN");
        cached = env ? strtoul(env, NULL, 10) : 64 * 1024;
    }
    return cached;
}

static void logCopy(const char *fn, void *dst, const void *src, size_t n) {
    if (n >= minBytes()) {
        fprintf(stderr, "[memcpy-trace] %s %zu bytes %p -> %p\n", fn, n, src, dst);
    }
}

static void *traced_memcpy(void *dst, const void *src, size_t n) {
    logCopy("memcpy", dst, src, n);
    return memmove(dst, src, n);
}

static void *traced_memmove(void *dst, const void *src, size_t n) {
    logCopy("memmove", dst, src, n);
    return memmove(dst, src, n);
}

DYLD_INTERPOSE(traced_memcpy, memcpy)
DYLD_INTERPOSE(traced_memmove, memmove)
