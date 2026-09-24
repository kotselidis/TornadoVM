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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.Test;

/**
 * Behaviour of the handle table the CUDA backend's Java layer addresses its state through: what
 * resolves, what does not, and that recycled slots never let a stale handle reach a new owner.
 */
public class CUDAHandlesTest {

    private static final long FIRST_HANDLE = 0x1000_0000L;

    private record Payload(int owner, int sequence) {
    }

    @Test
    public void registerResolveRelease() {
        Payload payload = new Payload(0, 0);
        long handle = CUDAHandles.register(payload);

        assertTrue("handles must not look like a wait-list count or 'no handle'", handle > FIRST_HANDLE);
        assertSame(payload, CUDAHandles.resolve(handle, Payload.class));
        assertSame(payload, CUDAHandles.resolve(handle, Object.class));
        assertNull("wrong type resolves to null", CUDAHandles.resolve(handle, String.class));

        assertSame(payload, CUDAHandles.release(handle));
        assertNull(CUDAHandles.resolve(handle, Payload.class));
        assertNull("double release", CUDAHandles.release(handle));
    }

    @Test
    public void staleHandleDoesNotReachRecycledSlot() {
        Payload first = new Payload(0, 1);
        long stale = CUDAHandles.register(first);
        CUDAHandles.release(stale);

        // The freed slot is recycled by the next registration, under a new generation.
        Payload second = new Payload(0, 2);
        long fresh = CUDAHandles.register(second);
        try {
            assertEquals("slot is recycled", stale & 0xFFFF_FFFFL, fresh & 0xFFFF_FFFFL);
            assertNotEquals(stale, fresh);
            assertNull(CUDAHandles.resolve(stale, Payload.class));
            assertNull(CUDAHandles.release(stale));
            assertSame("releasing the stale handle must not drop the new owner", second, CUDAHandles.resolve(fresh, Payload.class));
        } finally {
            CUDAHandles.release(fresh);
        }
    }

    @Test
    public void handlesThatWereNeverIssued() {
        for (long handle : new long[] { 0L, -1L, 1L, 42L, FIRST_HANDLE - 1, FIRST_HANDLE, FIRST_HANDLE + 12_345, Long.MAX_VALUE, Long.MIN_VALUE, 0x7FFF_FFFF_0000_0000L }) {
            assertNull("resolve " + Long.toHexString(handle), CUDAHandles.resolve(handle, Object.class));
            assertNull("release " + Long.toHexString(handle), CUDAHandles.release(handle));
        }
    }

    @Test
    public void manyLiveHandlesAcrossChunks() {
        int live = CUDAHandles.liveHandleCount();
        int n = 5_000;
        long[] handles = new long[n];
        Payload[] payloads = new Payload[n];
        for (int i = 0; i < n; i++) {
            payloads[i] = new Payload(1, i);
            handles[i] = CUDAHandles.register(payloads[i]);
        }
        assertEquals(live + n, CUDAHandles.liveHandleCount());
        for (int i = 0; i < n; i++) {
            assertSame(payloads[i], CUDAHandles.resolve(handles[i], Payload.class));
        }
        for (int i = 0; i < n; i += 2) {
            assertSame(payloads[i], CUDAHandles.release(handles[i]));
        }
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) {
                assertNull(CUDAHandles.resolve(handles[i], Payload.class));
            } else {
                assertSame(payloads[i], CUDAHandles.resolve(handles[i], Payload.class));
                CUDAHandles.release(handles[i]);
            }
        }
        assertEquals(live, CUDAHandles.liveHandleCount());
    }

    @Test
    public void registerReleaseCyclesDoNotGrowTheTable() {
        int live = CUDAHandles.liveHandleCount();
        for (int i = 0; i < 1_000_000; i++) {
            long h = CUDAHandles.register(this);
            CUDAHandles.release(h);
        }
        assertEquals(live, CUDAHandles.liveHandleCount());
    }

    /**
     * Several threads recycle slots as fast as they can while also resolving handles they have
     * already released. A resolve may only ever return the object registered under that exact
     * handle; a stale handle must resolve to null even while another thread owns the slot.
     */
    @Test
    public void concurrentRecyclingNeverLeaksAcrossOwners() throws Exception {
        int threads = 8;
        int rounds = 200_000;
        CountDownLatch start = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        List<Thread> workers = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
            final int owner = t;
            Thread worker = new Thread(() -> {
                try {
                    start.await();
                    long[] released = new long[64];
                    for (int i = 0; i < rounds && failure.get() == null; i++) {
                        Payload mine = new Payload(owner, i);
                        long h = CUDAHandles.register(mine);
                        Object seen = CUDAHandles.resolve(h, Object.class);
                        if (seen != mine) {
                            throw new AssertionError("live handle resolved to " + seen + " instead of " + mine);
                        }
                        long old = released[i & 63];
                        if (old != 0 && CUDAHandles.resolve(old, Object.class) != null) {
                            throw new AssertionError("stale handle " + Long.toHexString(old) + " resolved");
                        }
                        if (CUDAHandles.release(h) != mine) {
                            throw new AssertionError("release returned the wrong object");
                        }
                        released[i & 63] = h;
                    }
                } catch (Throwable e) {
                    failure.compareAndSet(null, e);
                }
            });
            workers.add(worker);
            worker.start();
        }
        start.countDown();
        for (Thread w : workers) {
            w.join();
        }
        if (failure.get() != null) {
            throw new AssertionError(failure.get());
        }
    }
}
