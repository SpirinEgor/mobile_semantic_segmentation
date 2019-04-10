package tech.spirin.segmentation.dnn

import android.content.res.AssetManager
import android.graphics.Bitmap
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


abstract class DNN(protected val assetManager: AssetManager) {

    abstract val assetsPath: String
    abstract val name: String
    abstract val inputShape: IntArray
    abstract val outputShape: IntArray

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