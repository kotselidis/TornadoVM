/*
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
package uk.ac.manchester.tornado.coreml.provider;

import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_INT;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_LONG;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_POINTER;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;

/**
 * Objective-C completion handlers through {@code java.lang.foreign}: a global block literal whose
 * invoke function is an upcall into Java. A global block is never copied to the heap, so the literal
 * stays where it is allocated until the handler has run.
 */
final class ObjCBlock {

    /** {@code BLOCK_IS_GLOBAL}: the block is static, {@code Block_copy} returns it unchanged. */
    private static final int BLOCK_IS_GLOBAL = 1 << 28;

    /** {@code struct Block_literal}: isa, flags, reserved, invoke, descriptor. */
    private static final MemoryLayout LITERAL = MemoryLayout.structLayout(C_POINTER.withName("isa"), C_INT.withName("flags"), C_INT.withName("reserved"), C_POINTER.withName("invoke"),
            C_POINTER.withName("descriptor"));

    /** {@code struct Block_descriptor}: reserved, size. */
    private static final MemoryLayout DESCRIPTOR = MemoryLayout.structLayout(C_LONG.withName("reserved"), C_LONG.withName("size"));

    private static final MemorySegment GLOBAL_BLOCK_ISA = globalBlockIsa();

    /** {@code void (^)(id object, NSError *error)}, as the invoke function sees it. */
    private static final FunctionDescriptor OBJECT_ERROR_HANDLER = FunctionDescriptor.ofVoid(C_POINTER, C_POINTER, C_POINTER);

    private static final MethodHandle COMPLETE;

    static {
        try {
            COMPLETE = MethodHandles.lookup().findStatic(ObjCBlock.class, "complete",
                    MethodType.methodType(void.class, CompletableFuture.class, MemorySegment.class, MemorySegment.class, MemorySegment.class));
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private ObjCBlock() {
    }

    private static MemorySegment globalBlockIsa() {
        MemorySegment isa = Linker.nativeLinker().defaultLookup().find("_NSConcreteGlobalBlock").orElse(null);
        if (isa == null && ObjC.isAvailable()) {
            var system = FFMSupport.loadLibrary("/usr/lib/libSystem.B.dylib");
            isa = system == null ? null : FFMSupport.findSymbol(system, "_NSConcreteGlobalBlock");
        }
        return isa;
    }

    static boolean isAvailable() {
        return GLOBAL_BLOCK_ISA != null;
    }

    /**
     * A completion handler {@code (id object, NSError *error)} to pass to {@code send}, and a wait for
     * it. The object is retained for the caller, who releases it.
     */
    static final class ObjectHandler {
        private final Arena arena = Arena.ofShared();
        private final CompletableFuture<Long> result = new CompletableFuture<>();
        private final MemorySegment literal;

        ObjectHandler() {
            if (!isAvailable()) {
                throw new TornadoRuntimeException("[ERROR] Objective-C blocks are not available");
            }
            MemorySegment invoke = FFMSupport.upcallStub(COMPLETE.bindTo(result), OBJECT_ERROR_HANDLER, arena);
            MemorySegment descriptor = arena.allocate(DESCRIPTOR);
            descriptor.set(C_LONG, 8, LITERAL.byteSize());
            literal = arena.allocate(LITERAL);
            literal.set(C_POINTER, 0, GLOBAL_BLOCK_ISA);
            literal.set(C_INT, 8, BLOCK_IS_GLOBAL);
            literal.set(C_POINTER, 16, invoke);
            literal.set(C_POINTER, 24, descriptor);
        }

        /** The block, as a message argument. */
        long block() {
            return literal.address();
        }

        /** Waits for the handler; returns the retained object or throws with the error it received. */
        long await(String what, long timeoutSeconds) {
            try {
                long object = result.get(timeoutSeconds, TimeUnit.SECONDS);
                arena.close();
                return object;
            } catch (ExecutionException e) {
                arena.close();
                throw new TornadoRuntimeException("[ERROR] " + what + ": " + e.getCause().getMessage());
            } catch (TimeoutException e) {
                // The handler may still run: its block and upcall stay allocated.
                throw new TornadoRuntimeException("[ERROR] " + what + ": no answer after " + timeoutSeconds + " s");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new TornadoRuntimeException("[ERROR] " + what + ": interrupted");
            }
        }
    }

    /** The invoke function of an {@link ObjectHandler}; runs on a Core ML queue and must not throw. */
    @SuppressWarnings("unused")
    private static void complete(CompletableFuture<Long> result, MemorySegment block, MemorySegment object, MemorySegment error) {
        try {
            if (object.address() != 0) {
                ObjC.retain(object.address());
                result.complete(object.address());
            } else {
                String message = error.address() == 0 ? "unknown error" : ObjC.string(ObjC.send(error.address(), "localizedDescription"));
                result.completeExceptionally(new TornadoRuntimeException(message));
            }
        } catch (Throwable t) {
            result.completeExceptionally(t);
        }
    }
}
