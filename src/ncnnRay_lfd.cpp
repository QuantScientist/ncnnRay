#include "../include/ncnnRay.hpp" // MUST BE INCLUDED FIRST


int main(int argc, char** argv) {
    VisionUtils vu = VisionUtils();
    vu.getGPU();

    std::string model_path = ".";
    bool useGPU = true;
//    bool useGPU = false;
    LFFD lffd1(model_path, 8, 0, useGPU);
    std::string fileName="faces.png";

    Image image = LoadImage(fileName.c_str());
    vu.detectFacesAndDrawOnImage(lffd1, image);
//    ExportImage(image, "faces-ncnn-rgb-gpu.png");

    return 0;
}
