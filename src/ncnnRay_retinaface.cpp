#include "../include/ncnnRay.hpp"
#include "models/FaceDetector.h"


using namespace std;

int main(int argc, char** argv)
{
    bool use_vulkan_compute = true;
    int gpu_device = 0;

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (use_vulkan_compute) {
        std::cout << "Using vulkan?: " << use_vulkan_compute << std::endl;

        g_vkdev = ncnn::get_gpu_device(gpu_device);
        g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);

        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    ncnn::Option opt = optGPU(use_vulkan_compute, gpu_device);
    std::string model_path = ".";
    std::string fileName = "faces01.png";

//    const int max_side = 320;

    // slim or RFB
    Detector detector (model_path, opt, false);
//    // scale
    Image img = LoadImage(fileName.c_str());   // Loaded in CPU memory (RAM)
    float long_side = std::max(img.width, img.height);
//    float scale = max_side/long_side;
//    ImageResize(&img, max_side, max_side);
    detector.detectFaces(img);
    ExportImage(img, "ncnn-rgb-retina.png");

    // draw image
//    for (int j = 0; j < boxes.size(); ++j) {
//
//        cv::Rect rect(boxes[j].x1/scale, boxes[j].y1/scale, boxes[j].x2/scale - boxes[j].x1/scale, boxes[j].y2/scale - boxes[j].y1/scale);
//        cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
//        char test[80];
//        sprintf(test, "%f", boxes[j].s);
//
//        cv::putText(img, test, cv::Size((boxes[j].x1/scale), boxes[j].y1/scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
//        cv::circle(img, cv::Point(boxes[j].point[0]._x / scale, boxes[j].point[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
//        cv::circle(img, cv::Point(boxes[j].point[1]._x / scale, boxes[j].point[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
//        cv::circle(img, cv::Point(boxes[j].point[2]._x / scale, boxes[j].point[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
//        cv::circle(img, cv::Point(boxes[j].point[3]._x / scale, boxes[j].point[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
//        cv::circle(img, cv::Point(boxes[j].point[4]._x / scale, boxes[j].point[4]._y / scale), 1, cv::Scalar(255, 0, 0), 4);
//    }
//    cv::imwrite("test.png", img);

    return 0;
}

