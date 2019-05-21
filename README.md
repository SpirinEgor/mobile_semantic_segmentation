# mobile_semantic_segmentation
Compare different neural network architecture for semantic segmentation problem on mobiles.

## Mobile app
Android application written in Kotlin and using tensorflow-mobile and tensorflow-lite.  
With this application you can load image and segmentate it with choosen Deep Neural Network or run benchmarks for compare average working time.

<p align="middle">
  <img src="/example_1.jpg" width="200" />
  <img src="/example_2.jpg" width="200" />
</p>

## Deep Neural Network
There are four networks, in this table you can find information about quality and size:

| DNN                      | mIoU  | Size (mb) |
|--------------------------|-------|-----------|
| DeepLab V3 CPU [513x513] | 0.797 | 8.8       |
| DeepLab V3 GPU [257x257] | 0.631 | 2.8       |
| U-Net CPU [224x224]      | 0.612 | 25.4      |
| IC-Net CPU [256x256]     | 0.639 | 27.1      |

And here you can find some time measurements:

|                          | Snapdragon 625 (4Gb RAM) | Snapdragon 845 (6Gb RAM) | Snapdragon 430 (4Gb RAM) | Exonys 7420 (3Gb RAM) |
|--------------------------|--------------------------|--------------------------|--------------------------|-----------------------|
| DeepLab V3 CPU [513x513] | 2307.75                  | 845.35                   | 5177.05                  | 2328.35               |
| DeepLab V3 GPU [257x257] | 391.6                    | 139.55                   | 630.85                   | 180.45                |
| U-Net CPU [224x224]      | 867.35                   | 414.68                   | 1933.13                  | 975.0                 |
| IC-Net CPU [256x256]     | 375.05                   | 127.6                    | 656.4                    | 248.85                |

You can find notebooks for each model with pretrained and trained weights, freezed graphs and `tflite` files. If you want, to run this notebooks, you should install packages from requirements, `tensorflow`, `keras` and this [repo](https://github.com/divamgupta/image-segmentation-keras). You can skip installation of frameworks and use my [docker](https://github.com/SpirinEgor/docker.jupyter).

## TODO
It's important to understand, that this project hasn't completed yet. So feel free to ask quiestions in issues.

- [ ] Implement E-Net (depends on Tensorflow ScatterNd layer [issue](https://github.com/tensorflow/tensorflow/issues/21526));
- [ ] Port IC-Net and U-Net to tflite;
- [ ] More networks??
- [ ] Train on MS COCO;
- [ ] ...
