package tech.spirin.segmentation.dnn

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import org.tensorflow.types.UInt8
import kotlin.math.roundToInt

class DeepLabV3CPU(assetManager: AssetManager) : DNN(assetManager) {

    //    override val assetsPath = "DeepLab_513_CPU/frozen_inference_graph.pb"
    override val assetsPath = "DeepLab_513_CPU/graph_dailystudio.pb"
    override val name = "DeepLab v3 513 CPU"
    override val inputShape = intArrayOf(1, 513, 513, 3)
    override val outputShape = intArrayOf(1, 513, 513)

    private lateinit var model: TensorFlowInferenceInterface

    private lateinit var inputData: ByteArray
    private lateinit var outputData: IntArray

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
        model = TensorFlowInferenceInterface(assetManager, assetsPath)
        inputData = ByteArray(inputShape[1] * inputShape[2] * inputShape[3])
        outputData = IntArray(outputShape[1] * outputShape[2])
    }

    override fun process(originalImage: Bitmap): Pair<Bitmap, Long> {
        val startProcess = System.currentTimeMillis()
        val originalHeight = originalImage.height
        val originalWidth = originalImage.width

        val scaledImage = Bitmap.createScaledBitmap(originalImage, inputShape[1], inputShape[2], false)
        val imageArray = IntArray(inputShape[1] * inputShape[2])
        scaledImage.getPixels(imageArray, 0, inputShape[2], 0, 0, inputShape[1], inputShape[2])

        for (i in imageArray.indices) {
            val pixel = imageArray[i]
            inputData[i * 3 + 0] = (pixel shr 16 and 0xFF).toByte()
            inputData[i * 3 + 1] = (pixel shr 8 and 0xFF).toByte()
            inputData[i * 3 + 2] = (pixel and 0xFF).toByte()
        }

        val start = System.currentTimeMillis()
        model.feed("ImageTensor", inputData, 1, 513, 513, 3)
        model.run(arrayOf("SemanticPredictions"))
        model.fetch("SemanticPredictions", outputData)
        val end = System.currentTimeMillis()

        val scaledMask = Bitmap.createBitmap(outputShape[1], outputShape[2], Bitmap.Config.ARGB_8888)
        for (x in 0 until outputShape[1]) {
            for (y in 0 until outputShape[2]) {
                val label = outputData[x + y * outputShape[2]]
                scaledMask.setPixel(x, y, labelColors[label].second)
            }
        }
        val finishProcess = System.currentTimeMillis()
        Log.i(this.name, "pre and post processing took ${finishProcess - startProcess - (end - start)} ms")
        return Bitmap.createScaledBitmap(scaledMask, originalWidth, originalHeight, true) to (end - start)
    }

}