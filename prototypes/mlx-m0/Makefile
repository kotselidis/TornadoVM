MLX_C ?= $(shell brew --prefix mlx-c)
BUILD := build

CFLAGS := -O2 -fobjc-arc -Wall -I$(MLX_C)/include
LDFLAGS := -L$(MLX_C)/lib -lmlxc -Wl,-rpath,$(MLX_C)/lib -framework Metal -framework Foundation

all: $(BUILD)/zerocopy_probe $(BUILD)/libmemcpy_trace.dylib

$(BUILD)/%: %.m
	@mkdir -p $(BUILD)
	clang $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD)/libmemcpy_trace.dylib: memcpy_trace.c
	@mkdir -p $(BUILD)
	clang -O2 -dynamiclib $< -o $@

# Zero-copy check without Instruments: log every memcpy/memmove >= 64 KiB.
trace: all
	DYLD_INSERT_LIBRARIES=$(BUILD)/libmemcpy_trace.dylib $(BUILD)/zerocopy_probe

run: all
	$(BUILD)/zerocopy_probe

clean:
	rm -rf $(BUILD)

.PHONY: all run trace clean
