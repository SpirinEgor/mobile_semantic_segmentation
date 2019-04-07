package tech.spirin.segmentation.dnn

import android.graphics.Bitmap

abstract class DNN {

    abstract val assetsPath: String
    abstract val name: String

    init {
        // Load dnn from assets
    }

    open fun process(originalImage: Bitmap) : Bitmap {
        return originalImage
    }

}