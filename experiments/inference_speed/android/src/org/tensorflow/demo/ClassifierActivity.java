/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.TypedValue;
import android.view.WindowManager;
import android.widget.Toast;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;


// public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
public class ClassifierActivity extends Activity {
    private static final Logger LOGGER = new Logger();

    protected static final boolean SAVE_PREVIEW_BITMAP = false;


    private ResultsView resultsView;

    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private long lastProcessingTimeMs;

    // These are the settings for the original v1 Inception model. If you want to
    // use a model that's been produced from the TensorFlow for Poets codelab,
    // you'll need to set IMAGE_SIZE = 299, IMAGE_MEAN = 128, IMAGE_STD = 128,
    // INPUT_NAME = "Mul", and OUTPUT_NAME = "final_result".
    // You'll also need to update the MODEL_FILE and LABEL_FILE paths to point to
    // the ones you produced.
    //
    // To use v3 Inception model, strip the DecodeJpeg Op from your retrained
    // model first:
    //
    // python strip_unused.py \
    // --input_graph=<retrained-pb-file> \
    // --output_graph=<your-stripped-pb-file> \
    // --input_node_names="Mul" \
    // --output_node_names="final_result" \
    // --input_binary=true
    private Handler handler;
    private HandlerThread handlerThread;

/*    // for classification
    private static final int INPUT_SIZE = 224;
    // for detection
    private static final int INPUT_WIDTH = 1920;
    private static final int INPUT_HEIGHT = 1080;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128;
    private static final String EXPERIMENT_MODE = "detection";
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "MobilenetV1/Predictions/Reshape_1";
    private static final String MODEL_FILE = "file:///android_asset/faster_rcnn_resnet101_v1.pb";
    private static final String LOG_FILE_NAME = "tf_faster_rcnn_resnet101v1_android.txt";
    private static final String LABEL_FILE =
            "file:///android_asset/imagenet_comp_graph_label_strings.txt";*/

    private static final boolean MAINTAIN_ASPECT = true;

    private Integer sensorOrientation;
    private Classifier classifier;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;


    private BorderedText borderedText;

    private int imageWidth;
    private int imageHeight;

    private File logFile;
    private BufferedWriter logBuf;
    private long totalTime;

    private volatile boolean isProcessing = false;

    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final String PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    private static final int PERMISSIONS_REQUEST = 1;
    private static final String LOG_TAG = "experiment";

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        LOGGER.d("onCreate " + this);
        super.onCreate(null);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_camera);

        final float textSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, 10, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);
        if (!hasPermission()) {
            requestPermission();
        }

        // start experiment
        for (Map.Entry<String, ExperimentConfig.ModelConfig> entry:
                ExperimentConfig.configs.entrySet()){
            Log.d(LOG_TAG, "Running experiment for " + entry.getKey());
            run_experiment(entry.getKey(), entry.getValue());
        }
    }

    private void run_experiment(String name, ExperimentConfig.ModelConfig config) {
        try {
            if (config.type == ExperimentConfig.ModelConfig.Type.CLASSIFIER){
                classifier =
                        TensorFlowImageClassifier.create(
                                getAssets(),
                                config.model_file,
                                config.label_file,
                                config.input_size,
                                config.image_mean,
                                config.image_std,
                                config.input_name,
                                config.output_name);
            } else if (config.type == ExperimentConfig.ModelConfig.Type.DETECTOR){
                classifier = TensorFlowObjectDetectionAPIModel.create(
                        getAssets(), config.model_file, config.label_file, config.input_size,
                        config.input_size);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // keep the matrix transformation just in case for imageWidth or height does not equal to
        // model input_size
        LOGGER.i("Initializing at size %dx%d", imageWidth, imageHeight);
        rgbFrameBitmap = Bitmap.createBitmap(imageWidth, imageHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(config.input_size, config.input_size, Config.ARGB_8888);

        sensorOrientation = 90;
        frameToCropTransform = ImageUtils.getTransformationMatrix(
                imageWidth, imageHeight,
                config.input_size, config.input_size,
                sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        String log_file_name = "tf_" + name + ".txt";
        logFile = new File(Environment.getExternalStorageDirectory().toString() + "/" + log_file_name);
        try {
            logFile.createNewFile();
            logBuf = new BufferedWriter(new FileWriter(logFile, false));
        } catch (IOException e) {
            e.printStackTrace();
        }
        infer();
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(PERMISSION_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA) ||
                    shouldShowRequestPermissionRationale(PERMISSION_STORAGE)) {
                Toast.makeText(this,
                        "Camera AND storage permission are required for this demo", Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[]{PERMISSION_CAMERA, PERMISSION_STORAGE}, PERMISSIONS_REQUEST);
        }
    }


    @Override
    public synchronized void onResume() {
        LOGGER.d("onResume " + this);
        super.onResume();

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    @Override
    public synchronized void onStart() {
        LOGGER.d("onStart " + this);
        super.onStart();
    }

    @Override
    public synchronized void onPause() {
        LOGGER.d("onPause " + this);

        if (!isFinishing()) {
            LOGGER.d("Requesting finish");
            finish();
        }

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            LOGGER.e(e, "Exception!");
        }

        super.onPause();
    }

    @Override
    public synchronized void onStop() {
        LOGGER.d("onStop " + this);
        super.onStop();
    }

    @Override
    public synchronized void onDestroy() {
        LOGGER.d("onDestroy " + this);
        super.onDestroy();
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    private Bitmap loadImageFromStorage(File f) throws FileNotFoundException {
        Bitmap b = BitmapFactory.decodeStream(new FileInputStream(f));
        return b;
    }

    private void infer() {
        String path = Environment.getExternalStorageDirectory().toString() + "/sample-images";
        Log.d("Files", "Path: " + path);
        File directory = new File(path);
        File[] files = directory.listFiles();
        Log.d("Files", "Size: " + files.length);

        // warm up
        String file_name = files[0].getName();
        Log.d("classifieractivity", "jj warm up with:" + files[0].getName());
        String image_path = Environment.getExternalStorageDirectory().getAbsolutePath() + "/sample-images/" + file_name;
        isProcessing = true;
        Runnable task = processImage(image_path, false);
        while (isProcessing) {
            try {
                synchronized (task) {
                    task.wait(1000);
                }
            } catch (InterruptedException e) {
                Log.e("classifieractivity", "error in waiting classification to finish");
                e.printStackTrace();
            }
        }

        int processing_num = 0;
        for (int rep = 0; rep < 3; rep++) {
            for (int i = 0; i < files.length; i++) {
                file_name = files[i].getName();
                Log.d("classifieractivity", "jj FileName:" + files[i].getName());
                image_path = Environment.getExternalStorageDirectory().getAbsolutePath() + "/sample-images/" + file_name;
                isProcessing = true;
                task = processImage(image_path, true);
                while (isProcessing) {
                    try {
                        synchronized (task) {
                            task.wait(1000);
                        }
                    } catch (InterruptedException e) {
                        Log.e("classifieractivity", "error in waiting classification to finish");
                        e.printStackTrace();
                    }
                }
                processing_num += 1;
            }
        }
        try {
            logBuf.flush();
            logBuf.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        LOGGER.i("jj totalTime: %s ms, processed image num: %d", totalTime, processing_num);
    }

    protected Runnable processImage(String image_path, final boolean logTime) {
        File image_file;
        try {
            image_file = new File(image_path);
            rgbFrameBitmap = loadImageFromStorage(image_file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            Log.e("classifieractivity", "jj input image file not found");
        }

        if (rgbFrameBitmap == null) {
            System.exit(1);
        }

        Runnable task = new Runnable() {
            @Override
            public void run() {
                final long startTime = SystemClock.uptimeMillis();
                final Canvas canvas = new Canvas(croppedBitmap);
                canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
                final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                LOGGER.i("jj detection took: %s ms, results: %s", lastProcessingTimeMs, results);
                if (logTime) {
                    totalTime += lastProcessingTimeMs;
                    try {
                        logBuf.append(String.valueOf(lastProcessingTimeMs));
                        logBuf.newLine();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                synchronized (this) {
                    isProcessing = false;
                    notifyAll();
                }
            }
        };
        runInBackground(task);
        return task;
    }
}
