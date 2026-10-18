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

import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_CHAR;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_DOUBLE;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_LONG;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_POINTER;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;

/**
 * Objective-C messaging through {@code java.lang.foreign}: {@code objc_msgSend} with a fixed
 * prototype per argument count (all arguments and results are pointers or {@code NSInteger}s), and
 * the few Foundation objects Core ML takes (strings, numbers, arrays, dictionaries).
 */
final class ObjC {

    private static final SymbolLookup LIBOBJC = FFMSupport.loadLibrary("/usr/lib/libobjc.A.dylib");
    private static final SymbolLookup FOUNDATION = FFMSupport.loadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation");
    private static final MethodHandle SEL_REGISTER_NAME = downcall(FunctionDescriptor.of(C_LONG, C_POINTER), "sel_registerName");
    private static final MethodHandle OBJC_GET_CLASS = downcall(FunctionDescriptor.of(C_LONG, C_POINTER), "objc_getClass");
    private static final MethodHandle POOL_PUSH = downcall(FunctionDescriptor.of(C_LONG), "objc_autoreleasePoolPush");
    private static final MethodHandle POOL_POP = downcall(FunctionDescriptor.ofVoid(C_LONG), "objc_autoreleasePoolPop");

    private static final Map<FunctionDescriptor, MethodHandle> SENDS = new ConcurrentHashMap<>();
    private static final Map<String, Long> SELECTORS = new ConcurrentHashMap<>();
    private static final Map<String, Long> CLASSES = new ConcurrentHashMap<>();

    private ObjC() {
    }

    /** Whether the Objective-C runtime and Foundation are loaded. */
    static boolean isAvailable() {
        return LIBOBJC != null && FOUNDATION != null;
    }

    static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException e) {
            return e;
        }
        if (t instanceof Error e) {
            throw e;
        }
        return new TornadoRuntimeException(t instanceof Exception e ? e : new Exception(t));
    }

    private static MethodHandle downcall(FunctionDescriptor descriptor, String name) {
        return LIBOBJC == null ? null : FFMSupport.downcall(LIBOBJC, descriptor, name);
    }

    static long sel(String name) {
        return SELECTORS.computeIfAbsent(name, key -> {
            try (Arena arena = Arena.ofConfined()) {
                return (long) SEL_REGISTER_NAME.invokeExact(FFMSupport.allocateCString(arena, key));
            } catch (Throwable t) {
                throw rethrow(t);
            }
        });
    }

    /** The class with this name, or 0 if no loaded image defines it. */
    static long objcClass(String name) {
        Long cached = CLASSES.get(name);
        if (cached != null) {
            return cached;
        }
        long cls;
        try (Arena arena = Arena.ofConfined()) {
            cls = (long) OBJC_GET_CLASS.invokeExact(FFMSupport.allocateCString(arena, name));
        } catch (Throwable t) {
            throw rethrow(t);
        }
        if (cls != 0) {
            CLASSES.put(name, cls);
        }
        return cls;
    }

    private static MethodHandle msgSend(FunctionDescriptor descriptor) {
        return SENDS.computeIfAbsent(descriptor, key -> FFMSupport.downcall(LIBOBJC, key, "objc_msgSend"));
    }

    static long send(long receiver, String selector) {
        try {
            return (long) msgSend(FunctionDescriptor.of(C_LONG, C_LONG, C_LONG)).invokeExact(receiver, sel(selector));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static long send(long receiver, String selector, long argument) {
        try {
            return (long) msgSend(FunctionDescriptor.of(C_LONG, C_LONG, C_LONG, C_LONG)).invokeExact(receiver, sel(selector), argument);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static long send(long receiver, String selector, long first, long second) {
        try {
            return (long) msgSend(FunctionDescriptor.of(C_LONG, C_LONG, C_LONG, C_LONG, C_LONG)).invokeExact(receiver, sel(selector), first, second);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static long send(long receiver, String selector, long first, long second, long third) {
        try {
            return (long) msgSend(FunctionDescriptor.of(C_LONG, C_LONG, C_LONG, C_LONG, C_LONG, C_LONG)).invokeExact(receiver, sel(selector), first, second, third);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static long send(long receiver, String selector, long a, long b, long c, long d, long e, long f) {
        try {
            return (long) msgSend(FunctionDescriptor.of(C_LONG, C_LONG, C_LONG, C_LONG, C_LONG, C_LONG, C_LONG, C_LONG, C_LONG)).invokeExact(receiver, sel(selector), a, b, c, d, e, f);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** A message whose result is an Objective-C {@code BOOL}. */
    static boolean sendBoolean(long receiver, String selector, long argument) {
        try {
            return (byte) msgSend(FunctionDescriptor.of(C_CHAR, C_LONG, C_LONG, C_LONG)).invokeExact(receiver, sel(selector), argument) != 0;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** A message whose result is a {@code double}. */
    static double sendDouble(long receiver, String selector) {
        try {
            return (double) msgSend(FunctionDescriptor.of(C_DOUBLE, C_LONG, C_LONG)).invokeExact(receiver, sel(selector));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void retain(long object) {
        if (object != 0) {
            send(object, "retain");
        }
    }

    static void release(long object) {
        if (object != 0) {
            send(object, "release");
        }
    }

    /** {@code [[cls alloc] init]}, owned by the caller. */
    static long newObject(String cls) {
        return send(send(objcClass(cls), "alloc"), "init");
    }

    static long poolPush() {
        try {
            return (long) POOL_PUSH.invokeExact();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void poolPop(long pool) {
        try {
            POOL_POP.invokeExact(pool);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ---------------------------------------------------------------- Foundation objects (autoreleased)

    static long nsString(String value) {
        try (Arena arena = Arena.ofConfined()) {
            return send(objcClass("NSString"), "stringWithUTF8String:", FFMSupport.allocateCString(arena, value).address());
        }
    }

    static String string(long nsString) {
        if (nsString == 0) {
            return null;
        }
        return FFMSupport.readCString(MemorySegment.ofAddress(send(nsString, "UTF8String")));
    }

    static long nsNumber(long value) {
        return send(objcClass("NSNumber"), "numberWithLongLong:", value);
    }

    static long nsArray(long... objects) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment items = FFMSupport.allocateLongArray(arena, objects);
            return send(objcClass("NSArray"), "arrayWithObjects:count:", items.address(), objects.length);
        }
    }

    static long nsNumbers(long[] values) {
        long[] numbers = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            numbers[i] = nsNumber(values[i]);
        }
        return nsArray(numbers);
    }

    static long nsDictionary(long[] keys, long[] values) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment k = FFMSupport.allocateLongArray(arena, keys);
            MemorySegment v = FFMSupport.allocateLongArray(arena, values);
            return send(objcClass("NSDictionary"), "dictionaryWithObjects:forKeys:count:", v.address(), k.address(), keys.length);
        }
    }

    static long count(long collection) {
        return send(collection, "count");
    }

    static long objectAt(long array, long index) {
        return send(array, "objectAtIndex:", index);
    }

    /** An {@code NSArray<NSNumber>} as longs. */
    static long[] longs(long array) {
        if (array == 0) {
            return new long[0];
        }
        long[] values = new long[(int) count(array)];
        for (int i = 0; i < values.length; i++) {
            values[i] = send(objectAt(array, i), "longLongValue");
        }
        return values;
    }

    /** The description of the {@code NSError} an out-parameter received, or a fallback. */
    static String errorMessage(MemorySegment errorSlot) {
        long error = errorSlot.get(C_POINTER, 0).address();
        if (error == 0) {
            return "unknown error";
        }
        return string(send(error, "localizedDescription"));
    }
}
