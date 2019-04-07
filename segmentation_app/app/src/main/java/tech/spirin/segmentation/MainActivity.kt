package tech.spirin.segmentation

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.support.v7.app.AlertDialog
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.widget.*
import tech.spirin.segmentation.dnn.DNN
import tech.spirin.segmentation.dnn.DeepLabV3GPU
import java.io.IOException
import java.util.*




class MainActivity : AppCompatActivity() {

    private lateinit var loadImageButton: Button
    private lateinit var processImageButton: Button
    private lateinit var originalImageView: ImageView
    private lateinit var processedImageView: ImageView
    private lateinit var blendedImageView: ImageView
    private lateinit var spinner: Spinner
    private lateinit var timeTextView: TextView

    private var currentImage: Bitmap? = null

    private val RESULT_GALLERY = 1
    private val RESULT_CAMERA = 2

    private lateinit var availableDNN: Array<DNN>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        requestMultiplePermissions(this)

        loadImageButton = findViewById(R.id.load_image)
        processImageButton = findViewById(R.id.process_image)
        originalImageView = findViewById(R.id.original_image)
        processedImageView = findViewById(R.id.processed_image)
        blendedImageView = findViewById(R.id.blended_image)
        spinner = findViewById(R.id.spinner)
        timeTextView = findViewById(R.id.time)

        availableDNN = arrayOf(DeepLabV3GPU(this.assets))
        availableDNN.map { it.initialize() }

        loadImageButton.setOnClickListener {
            showSelectImageDialog()
        }
        processImageButton.setOnClickListener {
            if (currentImage == null) {
                Toast.makeText(this, "Select image before process it", Toast.LENGTH_SHORT).show()
            } else {
                val selectedDNN = spinner.selectedItemPosition
                val result = availableDNN[selectedDNN].process(currentImage!!)
                processedImageView.setImageBitmap(result.first)
                timeTextView.text = "${result.second} ms"
                val blendImage = blendImages(currentImage!!, result.first)
                blendedImageView.setImageBitmap(blendImage)
            }
        }

        val spinnerAdapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_item, availableDNN.map { it.name }
        )
        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinner.adapter = spinnerAdapter
    }

    private fun showSelectImageDialog() {
        val dialog = AlertDialog.Builder(this)
        dialog.setTitle(getString(R.string.choose_image_source))
        val dialogItems = arrayOf(
            getString(R.string.select_from_gallery),
            getString(R.string.capture_from_camera)
        )
        dialog.setItems(dialogItems) { _, which ->
            when (which) {
                0 -> choosePhotoFromGallery()
                1 -> takePhotoFromCamera()
            }
        }
        dialog.show()
    }

    private fun choosePhotoFromGallery() {
        Log.i("Dialog", getString(R.string.select_from_gallery))
        val galleryIntent = Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(galleryIntent, RESULT_GALLERY)
    }

    private fun takePhotoFromCamera() {
        Log.i("Dialog", getString(R.string.capture_from_camera))
        val cameraIntent = Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(cameraIntent, RESULT_CAMERA)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == Activity.RESULT_CANCELED) {
            return
        }
        when (requestCode) {
            RESULT_CAMERA -> {
                Log.d("Get image", "result for capture photo")
                currentImage = data?.extras?.get("data") as Bitmap
                setOriginalImage(currentImage!!)
                val photoName = "${getString(R.string.app_name)}_${Calendar.getInstance().time}"
                val photoDescription = "Photo from ${getString(R.string.app_name)} application"
                val savedPhoto = MediaStore.Images.Media.insertImage(
                    this.contentResolver, currentImage, photoName, photoDescription
                )
                Log.i("Camera", "Save image to ${Uri.parse(savedPhoto)}")
            }
            RESULT_GALLERY -> {
                Log.d("Get image", "result for load from gallery")
                if (data != null) {
                    val contentURI = data.data
                    try {
                        currentImage = MediaStore.Images.Media.getBitmap(this.contentResolver, contentURI)
                        setOriginalImage(currentImage!!)
                    } catch (e: IOException) {
                        e.printStackTrace()
                        Toast.makeText(this, "Error while loading from gallery", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    private fun setOriginalImage(image: Bitmap) {
        originalImageView.setImageBitmap(image)
        processedImageView.setImageBitmap(null)
    }

}
