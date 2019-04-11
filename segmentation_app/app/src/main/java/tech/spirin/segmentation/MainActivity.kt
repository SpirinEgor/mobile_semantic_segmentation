package tech.spirin.segmentation

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.support.v4.content.FileProvider
import android.support.v7.app.AlertDialog
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.view.View
import android.widget.*
import tech.spirin.segmentation.dnn.DNN
import tech.spirin.segmentation.dnn.DeepLabV3CPU
import tech.spirin.segmentation.dnn.DeepLabV3GPU
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*


class MainActivity : AppCompatActivity() {

    private lateinit var loadImageButton: Button
    private lateinit var benchmarkButton: Button
    private lateinit var originalImageView: ImageView
    private lateinit var processedImageView: ImageView
    private lateinit var blendedImageView: ImageView
    private lateinit var description: TextView
    private lateinit var scrollView: ScrollView
    private lateinit var spinner: Spinner

    // using for taking photo and save it to gallery
    private var currentPhotoPath: String? = null
    private var currentImage: Bitmap? = null

    private val RESULT_GALLERY = 1
    private val RESULT_CAMERA = 2
    private val RESULT_ASSETS = 3

    private lateinit var availableDNN: Array<DNN>

    // asset images
    private lateinit var assetsImages: Array<String>
    private lateinit var assetsAdapter: SimpleAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        requestMultiplePermissions(this)

        loadImageButton = findViewById(R.id.load_image)
        benchmarkButton = findViewById(R.id.benchmark)
        originalImageView = findViewById(R.id.original_image)
        processedImageView = findViewById(R.id.processed_image)
        blendedImageView = findViewById(R.id.blended_image)
        description = findViewById(R.id.description)
        scrollView = findViewById(R.id.scroll_view)
        spinner = findViewById(R.id.spinner)

        availableDNN = arrayOf(DeepLabV3GPU(this.assets), DeepLabV3CPU(this.assets))
        availableDNN.map { it.initialize() }

        assetsImages = assets.list("images")!!
            .filter { !it.endsWith(".png") }
            .toTypedArray()
        val data = arrayListOf<HashMap<String, Any>>()
        assetsImages.forEach {
            val classes = it.split("_")
            val cur = hashMapOf(
                "image" to resources.getIdentifier(it, "drawable", "tech.spirin.segmentation"),
                "name" to classes.slice(0 until classes.size - 2).joinToString("\n")
            )
            data.add(cur)
        }
        val from = arrayOf("image", "name")
        val to = intArrayOf(R.id.dialog_image_view, R.id.dialog_image_name)
        assetsAdapter = SimpleAdapter(this, data, R.layout.image_dialog, from, to)

        loadImageButton.setOnClickListener {
            showSelectImageDialog()
        }

        benchmarkButton.setOnClickListener {
            description.text = "Measure time and IoU for all preloaded images...\n (TODO)"
            scrollView.visibility = View.INVISIBLE
        }

        val spinnerAdapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_item, availableDNN.map { it.name }
        )
        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinner.adapter = spinnerAdapter
    }

    private fun showSelectImageDialog() {
        val dialogBuilder = AlertDialog.Builder(this)
        dialogBuilder.setTitle(getString(R.string.choose_image_source))
        val dialogItems = arrayOf(
            getString(R.string.select_from_gallery),
            getString(R.string.capture_from_camera),
            getString(R.string.use_preload_image)
        )
        dialogBuilder.setItems(dialogItems) { _ , which ->
            when (which) {
                0 -> choosePhotoFromGallery()
                1 -> takePhotoFromCamera()
                2 -> loadFromAssets()
            }
        }
        dialogBuilder.show()
    }

    private fun choosePhotoFromGallery() {
        Log.i("Dialog", getString(R.string.select_from_gallery))
        val galleryIntent = Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(galleryIntent, RESULT_GALLERY)
    }

    private fun takePhotoFromCamera() {
        Log.i("Dialog", getString(R.string.capture_from_camera))
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            // Ensure that there's a camera activity to handle the intent
            takePictureIntent.resolveActivity(packageManager)?.also {
                // Create the File where the photo should go
                val photoFile: File? = try {
                    createImageFile()
                } catch (ex: IOException) {
                    // Error occurred while creating the File
                    ex.printStackTrace()
                    Toast.makeText(this, "Error while taking photo", Toast.LENGTH_SHORT).show()
                    null
                }
                // Continue only if the File was successfully created
                photoFile?.also {
                    val photoURI: Uri = FileProvider.getUriForFile(
                        this,
                        "com.example.android.fileprovider",
                        it
                    )
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    startActivityForResult(takePictureIntent, RESULT_CAMERA)
                }
            }
        }
    }

    private fun loadFromAssets() {
        val dialog = AlertDialog.Builder(this)
        dialog.setTitle(getString(R.string.choose_image_source))
        dialog.setAdapter(assetsAdapter) { _, which ->
            val intent = Intent().putExtra("assetId", which)
            onActivityResult(RESULT_ASSETS, Activity.RESULT_OK, intent)
        }
        dialog.show()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == Activity.RESULT_CANCELED) {
            return
        }
        when (requestCode) {
            RESULT_CAMERA -> {
                Log.d("Get image", "result for capture photo")
                val f = File(currentPhotoPath)
                val contentUri = Uri.fromFile(f)
                currentImage = MediaStore.Images.Media.getBitmap(this.contentResolver, contentUri)
            }
            RESULT_GALLERY -> {
                Log.d("Get image", "result for load from gallery")
                if (data != null) {
                    val contentUri = data.data
                    try {
                        currentImage = MediaStore.Images.Media.getBitmap(this.contentResolver, contentUri)
                    } catch (e: IOException) {
                        e.printStackTrace()
                        Toast.makeText(this, "Error while loading from gallery", Toast.LENGTH_SHORT).show()
                    }
                }
            }
            RESULT_ASSETS -> {
                if (data != null) {
                    val assetId = data.getIntExtra("assetId", 0)
                    val ims = assets.open("images/${assetsImages[assetId]}/image.png")
                    currentImage = BitmapFactory.decodeStream(ims)
                }
            }
        }
        if (currentImage == null) {
            Toast.makeText(this, "Select image before segment it", Toast.LENGTH_SHORT).show()
        } else {
            val selectedDNN = spinner.selectedItemPosition
            val result = availableDNN[selectedDNN].process(currentImage!!)
            scrollView.visibility = View.VISIBLE
            originalImageView.setImageBitmap(currentImage)
            processedImageView.setImageBitmap(result.first)
            blendedImageView.setImageBitmap(blendImages(currentImage!!, result.first))
            description.text = getString(R.string.time_measure, result.second)
        }
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        // Create an image file name
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "segmentation_${timeStamp}_", /* prefix */
            ".jpg", /* suffix */
            storageDir /* directory */
        ).apply {
            currentPhotoPath = absolutePath
        }
    }

}
