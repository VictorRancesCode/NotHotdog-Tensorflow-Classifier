package com.codigopanda.nothotdog_tensorflow_classifier;

/**
 * Created by victor on 7/18/17.
 */
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;
import com.codigopanda.nothotdog_tensorflow_classifier.util.Recognition;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

public class TensorFlowImageClassifier {
    private static final String TAG = "ImageClassifier";
    private static final int MAX_RESULTS = 2;
    private static final float THRESHOLD = 0.2f;
    private String inputName;
    private String outputName;
    private int inputSize;
    private int imageMean;
    private float imageStd;
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    public TensorFlowImageClassifier(AssetManager assetManager, String modelFilename, String labelFilename, int inputSize, int imageMean, float imageStd, String inputName, String outputName) {

        this.inputName = inputName;
        this.outputName = outputName;
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(assetManager.open(labelFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                this.labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!", e);
        }

        this.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
        final Operation operation = this.inferenceInterface.graphOperation(outputName);
        final int numClasses = (int) operation.output(0).shape().size(1);
        Log.i(TAG, "Read " + this.labels.size() + " labels, output layer size is " + numClasses);
        this.inputSize = inputSize;
        this.imageMean = imageMean;
        this.imageStd = imageStd;
        this.outputNames = new String[]{outputName};
        this.intValues = new int[inputSize * inputSize];
        this.floatValues = new float[inputSize * inputSize * 3];
        this.outputs = new float[numClasses];

    }


    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }
        Trace.endSection();

        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
        Trace.endSection();

        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();
        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        3,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });
        for (int i = 0; i < outputs.length; ++i) {

            if (outputs[i] > THRESHOLD) {
                pq.add(
                        new Recognition(
                                "" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], null));
            }
        }
        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection();
        return recognitions;
    }
}