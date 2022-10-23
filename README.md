# License-Plate-Recognition-YOLOv7-and-CNN
This project is for the ultimate manner of identifying the License Plate. Combining YOLOv7 object detection, Hough transform alignment, and CNN character recognition
	

**English below**

Chương trình nhận dạng biển số xe trong kho bãi, được dùng cho biển số xe Việt Nam cả 1 và 2 hàng. Sử dụng xử lý ảnh OpenCV và thuật toán KNN. Chi tiết mình sẽ làm một video youtube cập nhật sau.

This project using the machine learning method called KNN and OpenCV, which is a powerful library for image processing for recognising the Vietnamese license plate in the parking lot. The detail would be in the youtube link below: 

HOW TO USE:
* To test on image, run `Image_test2.py`. Remember to change the path of image in `data/image/`
* To test on video, run `Video_test2.py`. Remeber to record the video with size 1920x1080 
* Use `GenData.py` to generate KNN data points which is `classifications.txt` and `flattened_images.txt`
* `training_chars.png` is the input of `GenData.py`
* `Preprocess.py` contains functions for image processing
* Remember to set up neccesary libraries in `requirements.txt` 

Các bạn có thể tìm hiểu thêm tại [LINK YOUTUBE:](https://youtu.be/7erlCp6d5w8)

More about this project on [YOUTUBE:](https://youtu.be/7erlCp6d5w8)

Đọc file `Nhận diện biển số xe.docx` để biết thêm lý thuyết.

For more information, please download the `Nhận diện biển số xe.docx` file

## CÁC BƯỚC CHÍNH TRONG CỦA 1 BÀI TOÁN NHẬN DẠNG BIỂN SỐ XE
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
