/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.hailo;

import org.opencv.core.Point;
import org.opencv.core.Rect2d;

/** Class containing JNI for using the HAILO V8 over PCIE. */
public class HailoJNI {
    /** A class representing the result of a detection. */
    public static class HailoResult {
        /**
         * Create a HailoResult with the specified bounding box coordinates, confidence, and class ID.
         *
         * @param x1 The x coordinate of a vertex of the bounding box.
         * @param y1 The y coordinate of a vertex of the bounding box.
         * @param x2 The x coordinate of the opposite vertex of the bounding box.
         * @param y2 The y coordinate of the opposite vertex of the bounding box.
         * @param conf The confidence score of the detection.
         * @param class_id The class ID of the detected object.
         */
        public HailoResult(int x1, int y1, int x2, int y2, float conf, int class_id) {
            this.conf = conf;
            this.class_id = class_id;
            this.rect = new Rect2d(new Point(x1, y1), new Point(x2, y2));
        }

        /** Rectangle of the bounding box */
        public final Rect2d rect;

        /** Detection confidence */
        public final float conf;

        /** Class of the detected object */
        public final int class_id;

        @Override
        public String toString() {
            return "HailoResult [rect=" + rect + ", conf=" + conf + ", class_id=" + class_id + "]";
        }

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            result = prime * result + ((rect == null) ? 0 : rect.hashCode());
            result = prime * result + Float.floatToIntBits(conf);
            result = prime * result + class_id;
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null) return false;
            if (getClass() != obj.getClass()) return false;
            HailoResult other = (HailoResult) obj;
            if (rect == null) {
                if (other.rect != null) return false;
            } else if (!rect.equals(other.rect)) return false;
            if (Float.floatToIntBits(conf) != Float.floatToIntBits(other.conf)) return false;
            if (class_id != other.class_id) return false;
            return true;
        }
    }

    /**
     * Create a HailoJNI instance with the specified model path.
     *
     * @param modelPath Absolute path to the model file
     * @return A pointer to a struct with the hef detector instance.
     */
    public static native long create(String modelPath);

    /**
     * Destroy the HailoJNI instance.
     *
     * @param ptr The pointer to the hef detector instance.
     */
    public static native void destroy(long ptr);

    /**
     * Detect in the given image
     *
     * @param detectorPtr The pointer to the hef detector instance.
     * @param imagePtr The pointer to the image data.
     * @param boxThresh The threshold for the bounding box detection.
     * @param nmsThreshold The threshold for non-maximum suppression.
     * @return An array of {@link HailoJNI.HailoResult} objects containing the detection results.
     */
    public static native HailoResult[] detect(
            long detectorPtr, long imagePtr, double boxThresh, double nmsThreshold);
}
