#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <string.h>
#include <time.h>
#include <opencv2/opencv.hpp>

// Declare global variables
cl::Program program; // Program object

// Function to initialize OpenCL device and compile kernel code
void initializeDevice(cl::Context& context, cl::Device& device, cl::Program& program) {
    // Select the OpenCL platform
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Select the first platform
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
    context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

    // Get a list of devices on this platform
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (devices.empty()) {
        std::cerr << "No OpenCL devices found." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Select the first device
    device = devices[0];

    // Load kernel source code
    std::ifstream sourceFile("image_filtering.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    // Create program
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
    program = cl::Program(context, source);

    // Build program
    try {
        program.build("-cl-std=CL1.2");
    }
    catch (const cl::Error& e) {
        std::cerr << "Error building program: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        throw e;
    }
}

// Function to load OpenCL kernel code from file
std::string loadKernelCode(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open kernel file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    return std::string(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
}

// Function to create an OpenCL program from kernel code
cl::Program createProgram(const cl::Context& context, const cl::Device& device, const std::string& source) {
    cl::Program::Sources sources;
    sources.push_back({ source.c_str(), source.length() });
    cl::Program program(context, sources);
    if (program.build({ device }) != CL_SUCCESS) {
        std::cerr << "Error building program: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(EXIT_FAILURE);
    }
    return program;
}

// Function to apply grayscale conversion using OpenCL
void applyGrayscaleConversion(cv::Mat& inputImg, cl::Buffer& inputR, cl::Buffer& inputG, cl::Buffer& inputB, cl::Buffer& outputImg, cl::CommandQueue& queue) {
    cl::Kernel kernel(program, "rgb2gray");
    kernel.setArg(0, inputR);
    kernel.setArg(1, inputG);
    kernel.setArg(2, inputB);
    kernel.setArg(3, outputImg);
    cl::NDRange globalSize(inputImg.cols, inputImg.rows);
    cl::NDRange localSize(16, 16);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    queue.finish();
}

// Function to apply edge detection using OpenCL
void applyEdgeDetection(cv::Mat& inputImg, cl::Buffer& inputGrayImg, cl::Buffer& outputEdgeImg, cl::CommandQueue& queue) {
    cl::Kernel kernel(program, "edgeDetection");
    kernel.setArg(0, inputGrayImg);
    kernel.setArg(1, outputEdgeImg);
    cl::NDRange globalSize(inputImg.cols, inputImg.rows);
    cl::NDRange localSize(16, 16);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    queue.finish();
}

// Function to apply Gaussian blur using OpenCL
void applyGaussianBlur(cv::Mat& inputImg, cl::Buffer& inputEdgeImg, cl::Buffer& outputBlurImg, cl::CommandQueue& queue) {
    cl::Kernel kernel(program, "gaussianBlur");
    kernel.setArg(0, inputEdgeImg);
    kernel.setArg(1, outputBlurImg);
    cl::NDRange globalSize(inputImg.cols, inputImg.rows);
    cl::NDRange localSize(16, 16);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    queue.finish();
}

int main() {
    // Load input image using OpenCV
    cv::Mat inputImgMat = cv::imread("input_img.jpg");
    if (inputImgMat.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Extract R, G, B channels from the input image
    std::vector<cv::Mat> channels;
    cv::split(inputImgMat, channels);
    cv::Mat inputRchannel = channels[2]; // OpenCV uses BGR ordering
    cv::Mat inputGchannel = channels[1];
    cv::Mat inputBchannel = channels[0];

    // Initialize OpenCL device
    cl::Context context;
    cl::Device device;
    cl::Program program;
    cl::CommandQueue queue(context, device);
    initializeDevice(context, device, program);

    // Load OpenCL kernel code
    std::string kernelCode = loadKernelCode("image_filtering.cl");

    // Create OpenCL program
    program = createProgram(context, device, kernelCode);

    // Define mask sizes for edge detection and Gaussian blur
    const int edgeDetectionMaskSize = 3;
    const int gaussianBlurMaskSize = 5;

    // Allocate memory for output images
    cv::Mat outputGrayImgMat(inputImgMat.rows, inputImgMat.cols, CV_8UC1);
    cv::Mat outputEdgeImgMat(inputImgMat.rows, inputImgMat.cols, CV_8UC1);
    cv::Mat outputBlurImgMat(inputImgMat.rows, inputImgMat.cols, CV_8UC1);
    cl::Buffer inputR(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * inputImgMat.total(), inputRchannel.data);
    cl::Buffer inputG(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * inputImgMat.total(), inputGchannel.data);
    cl::Buffer inputB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * inputImgMat.total(), inputBchannel.data);
    cl::Buffer outputGrayImg(context, CL_MEM_READ_WRITE, sizeof(uchar) * outputGrayImgMat.total());
    cl::Buffer outputEdgeImg(context, CL_MEM_READ_WRITE, sizeof(uchar) * outputEdgeImgMat.total());
    cl::Buffer outputBlurImg(context, CL_MEM_READ_WRITE, sizeof(uchar) * outputBlurImgMat.total());

    // Apply grayscale conversion
    applyGrayscaleConversion(inputImgMat, inputR, inputG, inputB, outputGrayImg, queue);

    // Apply edge detection
    applyEdgeDetection(inputImgMat, outputGrayImg, outputEdgeImg, queue);

    // Apply Gaussian blur
    applyGaussianBlur(inputImgMat, outputEdgeImg, outputBlurImg, queue);

    // Read output images from device to host
    queue.enqueueReadBuffer(outputGrayImg, CL_TRUE, 0, sizeof(uchar) * outputGrayImgMat.total(), outputGrayImgMat.data);
    queue.enqueueReadBuffer(outputEdgeImg, CL_TRUE, 0, sizeof(uchar) * outputEdgeImgMat.total(), outputEdgeImgMat.data);
    queue.enqueueReadBuffer(outputBlurImg, CL_TRUE, 0, sizeof(uchar) * outputBlurImgMat.total(), outputBlurImgMat.data);

    // Display output images using OpenCV
    cv::imwrite("greyscale_output.jpg", outputGrayImgMat);
    cv::imwrite("edgedetect_output.jpg", outputEdgeImgMat);
    cv::imwrite("gaussianblur_output.jpg", outputBlurImgMat);
    //cv::waitKey(0);

    return 0;
}
