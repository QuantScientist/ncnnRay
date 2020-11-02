#include "../include/ncnnRay.hpp"
#include "models/FaceDetector.h"
#include <omp.h>

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
    Detector detector (model_path, opt, false);
    Image img={0};
    #pragma omp parallel for num_threads(10)
    for (int i=0; i<10000;i++) {
        printf("thread is %d\n", omp_get_thread_num());
        img = LoadImage(fileName.c_str());   // Loaded in CPU memory (RAM)
        detector.detectFaces(img);
    }

    ExportImage(img, "ncnn-rgb-retina.png");
    return 0;
}

