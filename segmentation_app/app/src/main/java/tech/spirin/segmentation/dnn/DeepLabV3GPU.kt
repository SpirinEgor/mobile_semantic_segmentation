package tech.spirin.segmentation.dnn

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.experimental.GpuDelegate



class DeepLabV3GPU(assetManager: AssetManager) : DNN(assetManager) {

    override val assetsPath = "DeepLab_GPU/deeplabv3_257_mv_gpu.tflite"
    override val name = "DeepLab v3 GPU"
    override val inputShape = intArrayOf(1, 257, 257, 3)
    override val outputShape = intArrayOf(1, 257, 257, 21)

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
    }

    override fun process(originalImage: Bitmap): Pair<Bitmap, Long> {
        val originalHeight = originalImage.height
        val originalWidth = originalImage.width

        val scaledImage = Bitmap.createScaledBitmap(originalImage, inputShape[1], inputShape[2], false)

        val input = Array(1) {
            Array(inputShape[1]) { Array(inputShape[2]) { FloatArray(inputShape[3]) } }
        }
        val output = Array(1) {
            Array(outputShape[1]) { Array(outputShape[2]) { FloatArray(outputShape[3]) } }
        }

        for (i in 0 until inputShape[1]) {
            for (j in 0 until inputShape[2]) {
                val pixel = scaledImage.getPixel(i, j)
                input[0][i][j][0] = (Color.red(pixel).toFloat() - 128) / 128f
                input[0][i][j][1] = (Color.green(pixel).toFloat() - 128) / 128f
                input[0][i][j][2] = (Color.blue(pixel).toFloat() - 128) / 128f
            }
        }

        val start = System.currentTimeMillis()
        model.run(input, output)
        val end = System.currentTimeMillis()

        val scaledMask = Bitmap.createBitmap(outputShape[1], outputShape[2], Bitmap.Config.ARGB_8888)
        for (x in 0 until outputShape[1]) {
            for (y in 0 until outputShape[2]) {
                var argMax = 0
                for (label in 0 until labelColors.size) {
                    if (output[0][x][y][argMax] < output[0][x][y][label]) {
                        argMax = label
                    }
                }
                scaledMask.setPixel(x, y, labelColors[argMax].second)
            }
        }
        return Bitmap.createScaledBitmap(scaledMask, originalWidth, originalHeight, true) to (end - start)
    }

}