/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 */
package uk.ac.manchester.tornado.drivers.cuda.ffm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * The opaque {@code long} handles the CUDA backend's Java layer passes around, and the state each
 * one stands for.
 *
 * <p>
 * The JNI implementation this replaces boxed CUDA primitives inside {@code malloc}'d C structs and
 * handed their addresses to Java as longs. That worked, but the state those structs carried -- a
 * kernel's packed arguments, a program's source, build log and compiled image -- is Java state that
 * had been pushed across the boundary, and every handle was a raw pointer the Java side could get
 * wrong without the JVM noticing. Here the state stays in Java objects and the handle is a plain
 * counter: a stale or fabricated handle resolves to {@code null} instead of dereferencing whatever
 * happens to be at that address.
 *
 * <p>
 * Handles start above zero because the Java layer already treats {@code 0} as "no handle".
 *
 * <p>
 * The table is on the dispatch hot path: every kernel launch and every transfer resolves several
 * handles and registers and releases events. It therefore avoids allocating: a handle encodes a slot
 * index (low 32 bits, offset by {@link #FIRST_HANDLE}) and that slot's generation (high 32 bits).
 * Slots are recycled through a free list; releasing a slot bumps its generation, so a stale handle
 * to a recycled slot still resolves to {@code null}. {@link #resolve} is lock-free: it reads the
 * slot's generation, the object, then the generation again. {@link #register} and {@link #release}
 * are serialised on a lock. Slots live in fixed-size chunks that are allocated on demand and never
 * moved, so a concurrent reader never sees a stale copy of the table.
 */
public final class CUDAHandles {

    /**
     * Handles start well above zero for two reasons. The Java layer already treats {@code 0} as "no
     * handle", and the event wait list is passed in two different layouts -- a plain list of
     * handles, and one prefixed with an element count -- so a small integer reaching
     * {@link #resolve} has to be recognisable as a count word rather than resolving to whatever
     * object happens to have been registered first. No wait list is anywhere near this long.
     */
    private static final long FIRST_HANDLE = 0x1000_0000L;

    private static final int CHUNK_BITS = 10;
    private static final int CHUNK_SIZE = 1 << CHUNK_BITS;
    private static final int CHUNK_MASK = CHUNK_SIZE - 1;
    private static final int MAX_CHUNKS = 1 << 14;

    /** Live handles the table can hold at once (16M); slot indices stay far below 2^32 - FIRST_HANDLE. */
    static final int MAX_SLOTS = MAX_CHUNKS * CHUNK_SIZE;

    private static final AtomicReferenceArray<Chunk> CHUNKS = new AtomicReferenceArray<>(MAX_CHUNKS);

    private static final Object LOCK = new Object();

    /** Recycled slots (a stack), guarded by {@link #LOCK}. */
    private static int[] freeSlots = new int[CHUNK_SIZE];
    private static int freeCount;

    /** Next never-used slot, guarded by {@link #LOCK}. */
    private static int nextSlot;

    /**
     * A fixed block of slots. {@code generations[i]} is the generation a handle to slot {@code i}
     * must carry to resolve; it starts at 1 (so every handle is above {@code 2^32}) and is bumped on
     * every release, which invalidates all handles issued for the slot so far.
     */
    private static final class Chunk {
        private final AtomicIntegerArray generations = new AtomicIntegerArray(CHUNK_SIZE);
        private final AtomicReferenceArray<Object> objects = new AtomicReferenceArray<>(CHUNK_SIZE);

        Chunk() {
            for (int i = 0; i < CHUNK_SIZE; i++) {
                generations.setPlain(i, 1);
            }
        }
    }

    private CUDAHandles() {
    }

    private static long encode(int slot, int generation) {
        return ((long) generation << 32) | (FIRST_HANDLE + slot);
    }

    /** The slot a handle names, or {@code -1} if it cannot be one this table issued. */
    private static int slotOf(long handle) {
        long slot = (handle & 0xFFFF_FFFFL) - FIRST_HANDLE;
        return (slot >= 0 && slot < MAX_SLOTS && (handle >>> 32) > 0) ? (int) slot : -1;
    }

    /** Registers {@code object} and returns the handle the Java layer will address it by. */
    public static long register(Object object) {
        synchronized (LOCK) {
            int slot;
            if (freeCount > 0) {
                slot = freeSlots[--freeCount];
            } else {
                if (nextSlot == MAX_SLOTS) {
                    throw new IllegalStateException("CUDA handle table is full: " + MAX_SLOTS + " live handles");
                }
                slot = nextSlot++;
            }
            Chunk chunk = CHUNKS.get(slot >>> CHUNK_BITS);
            if (chunk == null) {
                chunk = new Chunk();
                CHUNKS.set(slot >>> CHUNK_BITS, chunk);
            }
            int index = slot & CHUNK_MASK;
            chunk.objects.set(index, object);
            return encode(slot, chunk.generations.get(index));
        }
    }

    /** Resolves a handle, or returns {@code null} if it was never registered or has been released. */
    @SuppressWarnings("unchecked")
    public static <T> T resolve(long handle, Class<T> type) {
        int slot = slotOf(handle);
        if (slot < 0) {
            return null;
        }
        Chunk chunk = CHUNKS.get(slot >>> CHUNK_BITS);
        if (chunk == null) {
            return null;
        }
        int index = slot & CHUNK_MASK;
        int generation = (int) (handle >>> 32);
        if (chunk.generations.get(index) != generation) {
            return null;
        }
        Object object = chunk.objects.get(index);
        // A release (and possibly a re-register) between the two generation reads means the object
        // just read may belong to the slot's next owner: treat the handle as released.
        if (chunk.generations.get(index) != generation) {
            return null;
        }
        return type.isInstance(object) ? (T) object : null;
    }

    /** Handles currently registered and not yet released. */
    static int liveHandleCount() {
        synchronized (LOCK) {
            return nextSlot - freeCount;
        }
    }

    /** Drops a handle. Returns what it referred to so the caller can release native resources. */
    public static Object release(long handle) {
        int slot = slotOf(handle);
        if (slot < 0) {
            return null;
        }
        synchronized (LOCK) {
            Chunk chunk = CHUNKS.get(slot >>> CHUNK_BITS);
            if (chunk == null) {
                return null;
            }
            int index = slot & CHUNK_MASK;
            int generation = (int) (handle >>> 32);
            if (chunk.generations.get(index) != generation) {
                return null;
            }
            // Invalidate first, then clear: a concurrent resolve either fails its generation check
            // or reads the object before it is cleared, exactly as with a map lookup racing remove.
            int next = generation + 1;
            chunk.generations.set(index, next > 0 ? next : 1);
            Object object = chunk.objects.getAndSet(index, null);
            if (freeCount == freeSlots.length) {
                freeSlots = Arrays.copyOf(freeSlots, freeSlots.length * 2);
            }
            freeSlots[freeCount++] = slot;
            return object;
        }
    }

    /** A CUDA device: the driver's {@code CUdevice} plus the ordinal it was enumerated at. */
    public record Device(int device, int ordinal) {
    }

    /** A CUDA context bound to one device. {@code context} is the raw {@code CUcontext}. */
    public record Context(long context, int device, int ordinal) {
    }

    /**
     * A command queue: one {@code CUstream} plus the context and device it belongs to.
     * {@code properties} carries the OpenCL-style queue property bits the Java layer created it
     * with, which is what tells the profiler whether the queue was asked to time its operations.
     */
    /**
     * A command queue: a CUDA stream plus the context it belongs to.
     *
     * <p>
     * Not a record, because it carries one piece of mutable state. {@code pending} says whether
     * anything has been enqueued on the stream since it was last drained, so that a
     * {@code cuStreamSynchronize} against a stream that is already empty can be skipped. A task
     * graph ends with a blocking read-back, a flush and a wait, and the last two synchronise a
     * stream that the read-back already drained.
     *
     * <p>
     * It is {@code volatile} rather than an {@code AtomicBoolean} because the transitions are a
     * plain set and a plain clear -- there is no read-modify-write to lose -- and every write is a
     * conservative "assume work is outstanding".
     */
    public static final class Queue {

        private final long stream;
        private final long context;
        private final int device;
        private final long properties;
        private volatile boolean pending;

        public Queue(long stream, long context, int device, long properties) {
            this.stream = stream;
            this.context = context;
            this.device = device;
            this.properties = properties;
        }

        public long stream() {
            return stream;
        }

        public long context() {
            return context;
        }

        public int device() {
            return device;
        }

        public long properties() {
            return properties;
        }

        /** Records that work may now be outstanding on the stream. Always safe to call. */
        public void markPending() {
            pending = true;
        }

        /** Records that the stream has been drained. Only ever called after a successful sync. */
        public void clearPending() {
            pending = false;
        }

        public boolean isPending() {
            return pending;
        }
    }

    /**
     * An event pair. {@code event} is the completion event used for waits, queries and
     * dependencies; {@code start} is an optional timestamp recorded before the operation so that
     * {@code cuEventElapsedTime(start, event)} yields the operation's device time. {@code start} is
     * {@code 0} for events that do not bracket a timed operation, such as markers and barriers, and
     * their elapsed time is reported as zero.
     */
    public record Event(long event, long start) {
    }

    /** A program: CUDA C source, the image NVRTC produced for it, and the module it was loaded as. */
    public static final class Program {

        /** Matches the OpenCL {@code CL_BUILD_SUCCESS} the cloned Java enum expects. */
        public static final int BUILD_SUCCESS = 0;
        /** Matches the OpenCL {@code CL_BUILD_ERROR} the cloned Java enum expects. */
        public static final int BUILD_ERROR = -2;
        /** Matches the OpenCL {@code CL_BUILD_NONE} the cloned Java enum expects. */
        public static final int BUILD_NONE = -1;

        public final long context;
        public final String source;

        /** The loadable module image: an NVRTC cubin, NVRTC PTX, or a pre-supplied binary. */
        public byte[] binary;
        public String log = "";
        public int buildStatus = BUILD_NONE;
        public long module;
        public boolean moduleLoaded;

        public Program(long context, String source, byte[] binary) {
            this.context = context;
            this.source = source;
            this.binary = binary == null ? new byte[0] : binary;
        }
    }

    /** A kernel: a {@code CUfunction} plus the argument blobs staged for its next launch. */
    public static final class Kernel {

        public final long function;
        public final long module;
        public final String name;

        /**
         * One byte blob per argument index, in the order {@code cuLaunchKernel} expects them. A
         * blob is replaced wholesale when the argument is set again, and the vector is grown to fit
         * the highest index the caller has used.
         */
        public final List<byte[]> arguments = new ArrayList<>();

        public Kernel(long function, long module, String name) {
            this.function = function;
            this.module = module;
            this.name = name;
        }

        public void setArgument(int index, byte[] value) {
            while (arguments.size() <= index) {
                arguments.add(null);
            }
            arguments.set(index, value);
        }
    }
}
