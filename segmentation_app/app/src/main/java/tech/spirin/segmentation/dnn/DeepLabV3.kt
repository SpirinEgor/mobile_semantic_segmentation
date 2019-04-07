package tech.spirin.segmentation.dnn

import android.graphics.Bitmap

class DeepLabV3: DNN() {

    override val assetsPath = ""
    override val name = "DeepLab v3"

    override fun process(originalImage: Bitmap) : Bitmap {
        return originalImage
    }

}