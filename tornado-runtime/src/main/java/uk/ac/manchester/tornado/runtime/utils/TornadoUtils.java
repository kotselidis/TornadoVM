/*
 * This file is part of Tornado: A heterogeneous programming framework: 
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2020, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * Linking this library statically or dynamically with other modules is
 * making a combined work based on this library. Thus, the terms and
 * conditions of the GNU General Public License cover the whole
 * combination.
 *
 * As a special exception, the copyright holders of this library give you
 * permission to link this library with independent modules to produce an
 * executable, regardless of the license terms of these independent
 * modules, and to copy and distribute the resulting executable under
 * terms of your choice, provided that you also meet, for each linked
 * independent module, the terms and conditions of the license of that
 * module. An independent module is a module which is not derived from
 * or based on this library. If you modify this library, you may extend
 * this exception to your version of the library, but you are not
 * obligated to do so. If you do not wish to do so, delete this
 * exception statement from your version.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Authors: Juan Fumero
 *
 */
package uk.ac.manchester.tornado.runtime.utils;

import uk.ac.manchester.tornado.api.enums.TornadoDeviceType;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;

import java.lang.annotation.Annotation;
import java.lang.reflect.Field;

public class TornadoUtils {

    public static int getSizeReduction(int inputSize, TornadoDeviceType where) {
        switch (where) {
            case CPU:
                // If it is executed on the CPU, we return the number of threads
                // of the current
                // CPU
                return Runtime.getRuntime().availableProcessors();
            case GPU:
                // size will be the number of work-groups on the GPU
                int size = 1;
                if (inputSize > 256) {
                    size = inputSize / 256;
                }
                return size;
            default:
                return 0;
        }
    }

    public static boolean hasAnnotatedField(Object wrapper, Class<? extends Annotation> annotation) {
        for (Field field : wrapper.getClass().getDeclaredFields()) {
            if (field.isAnnotationPresent(annotation)) {
                return true;
            }
        }
        return false;
    }

    public static Object getAnnotatedObjectFromField(Object wrapper, Class<? extends Annotation> annotation) {
        Field storageField = null;
        for (Field field : wrapper.getClass().getDeclaredFields()) {
            if (field.isAnnotationPresent(annotation)) {
                storageField = field;
            }
        }
        assert storageField != null;
        storageField.setAccessible(true);
        try {
            return storageField.get(wrapper);
        } catch (IllegalAccessException e) {
            throw new TornadoRuntimeException(e);
        }
    }

    public static Object getObjectFromField(Field field, Object obj) {
        try {
            field.setAccessible(true);
            return field.get(obj);
        } catch (IllegalAccessException e) {
            throw new TornadoRuntimeException(e);
        }
    }

}
