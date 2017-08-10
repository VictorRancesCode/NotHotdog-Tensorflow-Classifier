package com.codigopanda.nothotdog_tensorflow_classifier.util;

import android.graphics.RectF;

/**
 * Created by victor on 7/18/17.
 */


public class Recognition {

    public final String id;

    public final String title;
    public final Float confidence;
    public RectF location;

    public Recognition(
            final String id, final String title, final Float confidence, final RectF location) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.location = location;
    }


    public Float getConfidence() {
        return confidence;
    }


    @Override
    public String toString() {
        String resultString = "";
        if (id != null) {
            resultString += "[" + id + ".";
        }

        if (title != null) {
            resultString += title + " ";
        }

        if (confidence != null) {
            resultString += String.format("(%.1f%%) ", confidence * 100.0f);
        }

        if (location != null) {
            resultString += location + "]";
        }

        return resultString.trim();
    }
}