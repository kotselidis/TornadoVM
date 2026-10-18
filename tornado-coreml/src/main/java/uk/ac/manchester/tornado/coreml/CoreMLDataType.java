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
package uk.ac.manchester.tornado.coreml;

/**
 * Element types of Core ML multi-arrays ({@code MLMultiArrayDataType}) and the TornadoVM arrays that
 * hold them.
 */
public enum CoreMLDataType {

    /** {@code float}, held in a {@code FloatArray}. */
    FLOAT32(0x10000 | 32, 4),

    /** IEEE half precision, held in a {@code HalfFloatArray}. */
    FLOAT16(0x10000 | 16, 2),

    /** {@code double}, held in a {@code DoubleArray}. */
    FLOAT64(0x10000 | 64, 8),

    /** {@code int}, held in an {@code IntArray}. */
    INT32(0x20000 | 32, 4);

    private final long value;
    private final int byteSize;

    CoreMLDataType(long value, int byteSize) {
        this.value = value;
        this.byteSize = byteSize;
    }

    /** The {@code MLMultiArrayDataType} value. */
    public long value() {
        return value;
    }

    /** Bytes per element. */
    public int byteSize() {
        return byteSize;
    }

    /** The type with the given {@code MLMultiArrayDataType} value, or null for one TornadoVM does not hold. */
    public static CoreMLDataType of(long value) {
        for (CoreMLDataType type : values()) {
            if (type.value == value) {
                return type;
            }
        }
        return null;
    }
}
