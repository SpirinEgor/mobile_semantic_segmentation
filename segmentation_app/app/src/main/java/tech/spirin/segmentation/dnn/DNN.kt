package tech.spirin.segmentation.dnn

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


abstract class DNN(protected val assetManager: AssetManager) {

    abstract val assetsPath: String
    abstract val name: String
    abstract val inputShape: IntArray
    abstract val outputShape: IntArray

    protected val labelColors = arrayOf(
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

    open fun initialize() { }

    open fun process(originalImage: Bitmap) : Pair<Bitmap, Long> {
        return originalImage to 0
    }

    protected fun loadModelFile(assetManager: AssetManager, MODEL_FILE: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

}