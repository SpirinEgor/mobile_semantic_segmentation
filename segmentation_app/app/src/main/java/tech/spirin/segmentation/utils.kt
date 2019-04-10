package tech.spirin.segmentation

import android.Manifest
import android.app.Activity
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
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
    val result = Bitmap.createBitmap(background.width, background.height, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(result)
    val paint = Paint()
    paint.alpha = 127
    canvas.drawBitmap(background, 0f, 0f, paint)
    canvas.drawBitmap(foreground, 0f, 0f, paint)
    return result
}

fun mixColors(in1: Int, in2: Int): Int = (in1 + in2) / 2