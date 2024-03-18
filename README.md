# OpenCL Image Processing

This project demonstrates the implementation of three image processing techniques - grayscale conversion, edge detection, and Gaussian blur - using OpenCL and OpenCV.

## Overview

Image processing is a fundamental task in computer vision and graphics. This project showcases the implementation of three commonly used techniques:

1. **Grayscale Conversion:** Converts a color image into a black and white image by averaging the intensity values of its color channels. This simplifies the image by removing color information, making it easier to analyze and process further.

2. **Edge Detection:** Identifies the edges of objects in an image by detecting areas of rapid intensity change. This helps in segmenting objects from their backgrounds, facilitating tasks like object detection and recognition.

3. **Gaussian Blur:** Smoothens an image by applying a Gaussian filter, which convolves the image with a Gaussian kernel. This reduces noise and detail, resulting in a softer appearance and improved visual quality.

## Dependencies

- OpenCV: OpenCV is used for loading and displaying images, as well as splitting color channels.
- OpenCL: OpenCL is used for parallel processing of image data on compatible hardware devices, such as GPUs.

## Usage

1. Clone this repository to your local machine.
2. Place your input image in the project directory and update the file path in the `main.cpp` file.
3. Compile and run the `main.cpp` file.
4. The output images will be saved in the project directory as `greyscale_output.jpg`, `edgedetect_output.jpg`, and `gaussianblur_output.jpg`.

