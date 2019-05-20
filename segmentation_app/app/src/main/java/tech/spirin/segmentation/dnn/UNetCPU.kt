package tech.spirin.segmentation.dnn

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.experimental.GpuDelegate

class UNetCPU(assetManager: AssetManager) : DNN(assetManager) {

    override val assetsPath = "UNet_256/UNet_256x256.pb"
    override val name = "UNet 256 CPU"
    override val inputShape = intArrayOf(256, 256, 3)
    override val outputShape = intArrayOf(256, 256, 21)

    private val imageMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val imageStd = floatArrayOf(0.229f, 0.224f, 0.225f)
    private val imageMaxPixel = 255f

    private lateinit var model: TensorFlowInferenceInterface

    private lateinit var inputData: FloatArray
    private lateinit var outputData: FloatArray
    private lateinit var scaledImage: Bitmap
    private lateinit var imageArray: IntArray
    private lateinit var scaledMask: Bitmap

    override fun initialize() {
        model = TensorFlowInferenceInterface(assetManager, assetsPath)

        inputData = FloatArray(inputShape.fold(1, { acc, i -> acc * i }))

        outputData = FloatArray(outputShape.fold(1, { acc, i -> acc * i }))

        scaledMask = Bitmap.createBitmap(outputShape[0], outputShape[1], Bitmap.Config.ARGB_8888)
    }

    override fun process(originalImage: Bitmap): Pair<Bitmap, Long> {
        val startProcess = System.currentTimeMillis()
        val originalHeight = originalImage.height
        val originalWidth = originalImage.width

        scaledImage = Bitmap.createScaledBitmap(originalImage, inputShape[0], inputShape[1], false)
        imageArray = IntArray(inputShape[0] * inputShape[1])
        scaledImage.getPixels(imageArray, 0, inputShape[1], 0, 0, inputShape[1], inputShape[0])

        for (i in imageArray.indices) {
            val pixel = imageArray[i]
            inputData[3 * i] = processPixelChannel((pixel shr 16 and 0xFF).toFloat(), 0)
            inputData[3 * i + 1] = processPixelChannel((pixel shr 8 and 0xFF).toFloat(), 1)
            inputData[3 * i + 2] = processPixelChannel((pixel and 0xFF).toFloat(), 2)
        }

        model.feed("input_1", inputData, 1, 256, 256, 3)
        val start = System.currentTimeMillis()
        model.run(arrayOf("softmax/truediv"))
        val end = System.currentTimeMillis()
        model.fetch("softmax/truediv", outputData)

        var argMax: Int
        var valMax: Float
        var curVal: Float
        for (x in 0 until outputShape[0]) {
            for (y in 0 until outputShape[1]) {
                argMax = 0
                valMax = -1f
                for (label in 0 until labelColors.size) {
                    curVal = outputData[x * outputShape[1] * outputShape[2] + y * outputShape[2] + label]
                    if (curVal > valMax) {
                        argMax = label
                        valMax = curVal
                    }
                }
                scaledMask.setPixel(y, x, labelColors[argMax].second)
            }
        }
        val finishProcess = System.currentTimeMillis()
        Log.i(this.name, "pre and post processing took ${finishProcess - startProcess - (end - start)} ms")
        return Bitmap.createScaledBitmap(scaledMask, originalWidth, originalHeight, false) to (end - start)
    }

    private fun processPixelChannel(channelValue: Float, channelNumber: Int): Float {
        var newChannelValue = channelValue / imageMaxPixel
        newChannelValue -= imageMean[channelNumber]
        newChannelValue /= imageStd[channelNumber]
        return newChannelValue
    }

}