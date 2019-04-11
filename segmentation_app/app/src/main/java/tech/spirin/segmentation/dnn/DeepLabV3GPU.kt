package tech.spirin.segmentation.dnn

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.experimental.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

class DeepLabV3GPU(assetManager: AssetManager) : DNN(assetManager) {

    override val assetsPath = "DeepLab_257_GPU/deeplabv3_257_mv_gpu.tflite"
    override val name = "DeepLab v3 257 GPU"
    override val inputShape = intArrayOf(1, 257, 257, 3)
    override val outputShape = intArrayOf(1, 257, 257, 21)

    private val imageMean = 128f
    private val imageStd = 128f

    private lateinit var model: Interpreter

    private lateinit var inputData: ByteBuffer
    private lateinit var outputData: ByteBuffer
    private val bytesPerPoint = 4
    private lateinit var scaledImage: Bitmap
    private lateinit var imageArray: IntArray
    private lateinit var scaledMask: Bitmap

    private val labelColors = arrayOf(
        "background" to Color.rgb(0, 0, 0),
        "aeroplane" to Color.rgb(128, 0, 0),
        "bicycle" to Color.rgb(0, 128, 0),
        "bird" to Color.rgb(128, 128, 0),
        "boat" to Color.rgb(0, 0, 128),
        "bottle" to Color.rgb(128, 0, 128),
        "bus" to Color.rgb(0, 128, 128),
        "car" to Color.rgb(128, 128, 128),
        "cat" to Color.rgb(64, 0, 0),
        "chair" to Color.rgb(192, 0, 0),
        "cow" to Color.rgb(64, 128, 0),
        "diningtable" to Color.rgb(192, 128, 0),
        "dog" to Color.rgb(64, 0, 128),
        "horse" to Color.rgb(192, 0, 128),
        "motorbike" to Color.rgb(64, 128, 128),
        "person" to Color.rgb(192, 128, 128),
        "pottedplant" to Color.rgb(0, 64, 0),
        "sheep" to Color.rgb(128, 64, 0),
        "sofa" to Color.rgb(0, 192, 0),
        "train" to Color.rgb(128, 192, 0),
        "tv" to Color.rgb(0, 64, 128)
    )

    override fun initialize() {
        val delegate = GpuDelegate()
        val options = Interpreter.Options().addDelegate(delegate)
        model = Interpreter(loadModelFile(assetManager, assetsPath), options)

        inputData = ByteBuffer.allocateDirect(inputShape[1] * inputShape[2] * inputShape[3] * bytesPerPoint)
        inputData.order(ByteOrder.nativeOrder())
        outputData = ByteBuffer.allocateDirect(outputShape[1] * outputShape[2] * outputShape[3] * bytesPerPoint)
        outputData.order(ByteOrder.nativeOrder())

        scaledMask = Bitmap.createBitmap(outputShape[1], outputShape[2], Bitmap.Config.ARGB_8888)
    }

    override fun process(originalImage: Bitmap): Pair<Bitmap, Long> {
        val startProcess = System.currentTimeMillis()
        val originalHeight = originalImage.height
        val originalWidth = originalImage.width

        scaledImage = Bitmap.createScaledBitmap(originalImage, inputShape[1], inputShape[2], false)
        imageArray = IntArray(inputShape[1] * inputShape[2])
        scaledImage.getPixels(imageArray, 0, inputShape[2], 0, 0, inputShape[1], inputShape[2])

        inputData.rewind()
        outputData.rewind()
        for (i in imageArray.indices) {
            val pixel = imageArray[i]
            inputData.putFloat(((pixel shr 16 and 0xFF).toFloat() - imageMean) / imageStd)
            inputData.putFloat(((pixel shr 8 and 0xFF).toFloat() - imageMean) / imageStd)
            inputData.putFloat(((pixel and 0xFF).toFloat() - imageMean) / imageStd)
        }

        val start = System.currentTimeMillis()
        model.run(inputData, outputData)
        val end = System.currentTimeMillis()

        var argMax: Int
        var valMax: Float
        var curVal: Float
        for (y in 0 until outputShape[2]) {
            for (x in 0 until outputShape[1]) {
                argMax = 0
                valMax = -1f
                for (label in 0 until labelColors.size) {
                    curVal = outputData.getFloat(
                        (y * outputShape[2] * outputShape[3] + x * outputShape[3] + label) * bytesPerPoint
                    )
                    if (curVal > valMax) {
                        argMax = label
                        valMax = curVal
                    }
                }
                scaledMask.setPixel(x, y, labelColors[argMax].second)
            }
        }
        val finishProcess = System.currentTimeMillis()
        Log.i(this.name, "pre and post processing took ${finishProcess - startProcess - (end - start)} ms")
        return Bitmap.createScaledBitmap(scaledMask, originalWidth, originalHeight, false) to (end - start)
    }

}