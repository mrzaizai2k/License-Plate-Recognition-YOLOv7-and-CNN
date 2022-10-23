# License-Plate-Recognition-YOLOv7-and-CNN
This project is for the ultimate manner of identifying the License Plate. Combining YOLOv7 object detection, Hough transform alignment, and CNN character recognition
	
I created a [Vietnamese License Plate Recognition tool using image processing OpenCV and KNN algorithm](https://github.com/mrzaizai2k/VIETNAMESE_LICENSE_PLATE) (simple, but effective)

## Table of contents
* [1. How to use](#1-How-to-use)
* [2. User](#2-User)
* [3. Dataset](#3-Dataset)

## 1. How to use

**HOW TO USE:**
* Remember to set up neccesary libraries in `requirements.txt` 
* Download the model used for YOLOv7 model `LP_detect_yolov7_500img.pt` and CNN model `weight.h5` in Git RELEASES and put them in the right path like in the code
* To test on image/video, run `main_image.py`/ `main_video.py`. Remember to change the path of image/video. I don't provide videos for testing, but you can record it yourself. **1920x1080 pixels, 24 fps recommend**
* In `data` folder you can find `data.yaml` needed for YOLOv7 training and folder `test` including test images. Feel free to use it
* `doc` images for documents
* `src` folder are codes for CNN model. put the CNN model here
* `utils` and `models` are for YOLOv7. They're a part of original YOLO. However, don't care about them, you can use YOLOv7 to derectly detect License Plates with `detect.py`. I have changed the code a lot compared to the original one. It's now much easier to use
* `Preprocess.py`, `utils_LP.py` and `vid2img.py` are util files. Spend time to explore them.
* `yolo-v7-license-plate-detection.ipynb` is the training of YOLOv7

As you know:
**The main stages in the license plate recoginition algorithm**

1. License Plate Detection
2. Character Segmentation
3. Character Recognition

<p align="center"><img src="https://user-images.githubusercontent.com/40959407/130982072-a4701080-e40d-42c1-8fc5-062da340ca5b.png" width="300"></p>
<p align="center"><i>Figure 1. The main stages in the license plate recoginition algorithm </i></p>

## PHÁT HIỆN VÀ TÁCH BIỂN SỐ:
**The main stages in detecting and extract the license plate**
1. Taking picture from the camera
2. Gray scaling
3. Increasing the contrast level
4. Noise Decreasing by Gaussian filter
5. Adaptive threshold for image binarization
6. Canny Edge detection
7. Detect the plate by drawing contours and if..else
