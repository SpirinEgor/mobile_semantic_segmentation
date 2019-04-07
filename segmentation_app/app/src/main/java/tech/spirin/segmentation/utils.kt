package tech.spirin.segmentation

import android.Manifest
import android.app.Activity
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import android.widget.Toast
import com.karumi.dexter.Dexter
import com.karumi.dexter.MultiplePermissionsReport
import com.karumi.dexter.PermissionToken
import com.karumi.dexter.listener.PermissionRequest
import com.karumi.dexter.listener.multi.MultiplePermissionsListener
import java.nio.IntBuffer


// Listener for dexter, check granted permissions
class Listener(private val activity: Activity) : MultiplePermissionsListener {

    override fun onPermissionsChecked(report: MultiplePermissionsReport) {
        if (report.areAllPermissionsGranted()) {
            Log.i("Permissions", "All permissions granted")
        }
        if (report.isAnyPermissionPermanentlyDenied) {
            Toast.makeText(activity, "You should grant all permissions", Toast.LENGTH_LONG).show()
        }
    }

    override fun onPermissionRationaleShouldBeShown(
        permissions: MutableList<PermissionRequest>,
        token: PermissionToken
    ) {
        token.continuePermissionRequest()
    }

}

// Request needed permissions
fun requestMultiplePermissions(activity: Activity) {

    Dexter.withActivity(activity)
        .withPermissions(
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_EXTERNAL_STORAGE
        )
        .withListener(Listener(activity))
        .onSameThread()
        .check()
}

fun blendImages(background: Bitmap, foreground: Bitmap): Bitmap {
    val buffBase = IntBuffer.allocate(background.width * background.height)
    background.copyPixelsToBuffer(buffBase)
    buffBase.rewind()

    val buffBlend = IntBuffer.allocate(foreground.width * foreground.height)
    foreground.copyPixelsToBuffer(buffBlend)
    buffBlend.rewind()

    val buffOut = IntBuffer.allocate(background.width * background.height)
    buffOut.rewind()

    while (buffOut.position() < buffOut.limit()) {
        val filterInt = buffBlend.get()
        val srcInt = buffBase.get()

        val redValueFilter = Color.red(filterInt)
        val greenValueFilter = Color.green(filterInt)
        val blueValueFilter = Color.blue(filterInt)

        val redValueSrc = Color.red(srcInt)
        val greenValueSrc = Color.green(srcInt)
        val blueValueSrc = Color.blue(srcInt)

        val redValueFinal = mixColors(redValueFilter, redValueSrc)
        val greenValueFinal = mixColors(greenValueFilter, greenValueSrc)
        val blueValueFinal = mixColors(blueValueFilter, blueValueSrc)

        val pixel = Color.argb(255, redValueFinal, greenValueFinal, blueValueFinal)

        buffOut.put(pixel)
    }

    buffOut.rewind()

    val result = Bitmap.createBitmap(background.width, background.height, Bitmap.Config.ARGB_8888)
    result.copyPixelsFromBuffer(buffOut)
    return result
}

fun mixColors(in1: Int, in2: Int): Int = (in1 + in2) / 2