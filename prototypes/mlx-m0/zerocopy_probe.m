/*
 * M0 zero-copy probe: can mlx-c adopt memory that TornadoVM's Metal backend allocated?
 *
 * TornadoVM allocates one MTLBuffer per array with MTLResourceStorageModeShared
 * (MetalObjects.createBuffer) and places the data ARRAY_HEADER bytes into it
 * (16 with compressed oops, 24 without). This probe reproduces that layout,
 * wraps contents() with mlx_array_new_data_managed, skips the header with
 * as_strided, and reports whether MLX kept the pointer (zero-copy) or copied.
 *
 * Build: make -C prototypes/mlx-m0
 */
#import <Metal/Metal.h>
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int dtorCalls = 0;
static void countingDtor(void *p) { (void) p; dtorCalls++; }

static int failures = 0;
#define CHECK(cond, ...)                                                                                                                  \
    do {                                                                                                                                  \
        if (!(cond)) {                                                                                                                    \
            printf("  FAIL: " __VA_ARGS__);                                                                                               \
            printf("\n");                                                                                                                 \
            failures++;                                                                                                                   \
        }                                                                                                                                 \
    } while (0)

/*
 * Wraps a TornadoVM-style buffer of n floats behind a header, computes y = x * 2 + 1
 * on the GPU, and checks results, the zero-copy signals, and host-write visibility.
 */
static void probe(id<MTLDevice> dev, mlx_stream gpu, int n, int header, BOOL roundToPage) {
    size_t page = (size_t) getpagesize();
    size_t bytes = (size_t) header + (size_t) n * sizeof(float);
    size_t alloc = roundToPage ? (bytes + page - 1) / page * page : bytes;

    id<MTLBuffer> buf = [dev newBufferWithLength:alloc options:MTLResourceStorageModeShared];
    uint8_t *base = (uint8_t *) buf.contents;
    float *data = (float *) (base + header);
    for (int i = 0; i < n; i++) {
        data[i] = (float) i;
    }

    printf("n=%-8d header=%-2d alloc=%-9zu (%s) base %% page = %zu\n", n, header, alloc, roundToPage ? "page-rounded" : "exact", (size_t) base % page);

    // Wrap the whole allocation (header included) as a flat float array, then view past the header.
    dtorCalls = 0;
    int wholeShape[1] = { (int) (alloc / sizeof(float)) };
    mlx_array whole = mlx_array_new_data_managed(base, wholeShape, 1, MLX_FLOAT32, countingDtor);
    CHECK(dtorCalls == 0, "dtor ran right after wrapping -> MLX copied the buffer");
    int copiedAtWrap = dtorCalls != 0;

    mlx_array x = mlx_array_new();
    int shape[1] = { n };
    int64_t strides[1] = { 1 };
    mlx_as_strided(&x, whole, shape, 1, strides, 1, (size_t) header / sizeof(float), gpu);

    mlx_array two = mlx_array_new_float32(2.0f), one = mlx_array_new_float32(1.0f);
    mlx_array t = mlx_array_new(), y = mlx_array_new();
    mlx_multiply(&t, x, two, gpu);
    mlx_add(&y, t, one, gpu);
    mlx_array_eval(y);
    mlx_array_eval(whole);

    // Signal 1: after eval, does the wrapped array still point at TornadoVM's memory?
    const float *wholePtr = mlx_array_data_float32(whole);
    int samePtr = (const void *) wholePtr == (const void *) base;
    CHECK(samePtr, "wrapped array data %p != MTLBuffer contents %p", (const void *) wholePtr, (void *) base);

    const float *out = mlx_array_data_float32(y);
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (out[i] != 2.0f * i + 1.0f) {
            if (bad++ < 3) {
                printf("  y[%d]=%f expected %f\n", i, out[i], 2.0f * i + 1.0f);
            }
        }
    }
    CHECK(bad == 0, "%d wrong results", bad);

    // Signal 2: a host write after wrapping must be visible to a new MLX op on the same array.
    data[0] = 1000.0f;
    mlx_array y2 = mlx_array_new();
    mlx_add(&y2, x, one, gpu);
    mlx_array_eval(y2);
    float seen = mlx_array_data_float32(y2)[0];
    int hostWriteVisible = seen == 1001.0f;
    CHECK(hostWriteVisible, "host write not visible to MLX (saw %f) -> MLX is reading a copy", seen);

    printf("  results=%s  copiedAtWrap=%s  samePtr=%s  hostWriteVisible=%s  => %s\n", bad ? "WRONG" : "ok", copiedAtWrap ? "yes" : "no",
            samePtr ? "yes" : "no", hostWriteVisible ? "yes" : "no", (!copiedAtWrap && samePtr && hostWriteVisible) ? "ZERO-COPY" : "COPY");

    mlx_array_free(y2);
    mlx_array_free(y);
    mlx_array_free(t);
    mlx_array_free(one);
    mlx_array_free(two);
    mlx_array_free(x);
    mlx_array_free(whole);
    // MLX releases buffers from the command-buffer completion handler; wait for it.
    mlx_synchronize(gpu);
    CHECK(dtorCalls == 1, "dtor ran %d times after freeing the wrapped array (expected 1)", dtorCalls);
}

/* Step 1 of M0: basic load, matmul, eval and read-back. */
static void smoke(mlx_stream gpu) {
    float av[4] = { 1, 2, 3, 4 }, bv[4] = { 5, 6, 7, 8 };
    int shape[2] = { 2, 2 };
    mlx_array a = mlx_array_new_data(av, shape, 2, MLX_FLOAT32);
    mlx_array b = mlx_array_new_data(bv, shape, 2, MLX_FLOAT32);
    mlx_array c = mlx_array_new();
    mlx_matmul(&c, a, b, gpu);
    mlx_array_eval(c);
    const float *r = mlx_array_data_float32(c);
    int ok = r[0] == 19 && r[1] == 22 && r[2] == 43 && r[3] == 50;
    printf("smoke: 2x2 matmul = [%g %g; %g %g] %s\n", r[0], r[1], r[2], r[3], ok ? "ok" : "WRONG");
    CHECK(ok, "matmul smoke test");
    mlx_array_free(a);
    mlx_array_free(b);
    mlx_array_free(c);
}

int main(void) {
    @autoreleasepool {
        mlx_string v = mlx_string_new();
        mlx_version(&v);
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("mlx %s on %s, page size %d\n\n", mlx_string_data(v), dev.name.UTF8String, getpagesize());
        mlx_string_free(v);

        mlx_stream gpu = mlx_default_gpu_stream_new();
        smoke(gpu);
        printf("\n");

        int sizes[] = { 1, 1000, 4096, 4096 * 4096 };
        int headers[] = { 16, 24 };
        for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
            for (unsigned h = 0; h < 2; h++) {
                probe(dev, gpu, sizes[s], headers[h], NO);
                probe(dev, gpu, sizes[s], headers[h], YES);
            }
        }

        mlx_stream_free(gpu);
        printf("\n%s (%d failed checks)\n", failures ? "PROBE FOUND COPIES OR ERRORS" : "ALL ZERO-COPY", failures);
        return failures ? 1 : 0;
    }
}
