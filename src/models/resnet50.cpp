#include "resnet50.h"
#include <iostream>
#include <string>

FeatureExtractor::FeatureExtractor() {}

FeatureExtractor::~FeatureExtractor() {
    net.clear();
}

FeatureExtractor::FeatureExtractor(const std::string &model_path,
                                   const std::string &model_name,
                                   const ncnn::Option &opt
) {
    net.opt = opt;
#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute) {
        TraceLog(LOG_INFO, "ncnnRay: FeatureExtractor Opt using vulkan::%i", net.opt.use_vulkan_compute);
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN
    std::string param = model_path + "/" + model_name + ".param";
    std::string bin = model_path + "/" + model_name + ".bin";
    net.load_param(param.c_str());
    net.load_model(bin.c_str());
    TraceLog(LOG_INFO, "ncnnRay: FeatureExtractor model loaded, GPU enabled?=%i", net.opt.use_vulkan_compute);
    ncnn::VkMat blob_gpu;
}

std::vector<float> FeatureExtractor::ExtractFeature(Image &image,
                                                    const std::string &in_name, const std::string &out_name) {
    ScopeTimer Tmr("FeatureExtractor::ExtractFeature");

    ncnn::Mat in = rayImageToNcnn(image);

//    ncnn::VkMat in_gpu;
//    in_gpu.create_like(in, net.opt.blob_vkallocator);
//    const float mean_vals[3] = { 0.f, 0.f, 0.f };
//    const float norm_vals[3] = { 1.0 / 255, 1.0 / 255, 1.0 / 255 };
//    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = net.create_extractor();
//    #pragma omp parallel sections + #pragma omp section
//    #pragma omp parallel for num_threads(8)
//    for (int i=0; i<100; i++)
//    {
//        ncnn::Extractor ex = net1.create_extractor();
//        ex.input("data", inputs[i]);
//        ex.extract("prob", outputs[i]);
//    }

    ex.input(in_name.c_str(), in); // "input.1"
    ncnn::Mat out;
    ex.extract(out_name.c_str(), out); // Convolution Conv_116 1 1 488 652 0=2048 1=1 5=1
//    out = out.channel(0);
    TraceLog(LOG_INFO, "ncnnRay: out.c =%d", out.c);
    std::vector<float> vec;
    for (int i = 0; i < out.c; ++i) {
        vec.emplace_back(*(static_cast<float *>(out.data) + i));
    }
//    out=out.channel(0).reshape(2048);
//    std::vector<float> vec(2048);
//    memcpy(vec.data(), out, 2048*sizeof(float));

    TraceLog(LOG_INFO, "ncnnRay: vec =%d", vec.size());
    normalize(vec);
    return vec;
}

void FeatureExtractor::normalize(std::vector<float> &arr) {
    double mod = 0.0;
    for (float i : arr) {
        mod += i * i;
    }
    double mag = std::sqrt(mod);
    if (mag == 0) {
        throw std::logic_error("The input vector is a zero vector");
    }
    for (float &i : arr) {
        i /= mag;
    }
}

float FeatureExtractor::getSimilarity(const std::vector<float> &v1, const std::vector<float> &v2) {
    if (v1.size() != v2.size())
        throw std::invalid_argument("Wrong size");

    double mul = 0;
    for (size_t i = 0; i < v1.size(); ++i) {
        mul += v1[i] * v2[i];
    }

    if (mul < 0) {
        return 0;
    }
    return mul;
}

float FeatureExtractor::calculateSimilarity(const std::vector<float> &feat1, const std::vector<float> &feat2) {
    double dot = 0;
    double norm1 = 0;
    double norm2 = 0;
    for (size_t i = 0; i < feat1.size(); ++i) {
        dot += feat1[i] * feat2[i];
        norm1 += feat1[i] * feat1[i];
        norm2 += feat2[i] * feat2[i];
    }
    return dot / (sqrt(norm1 * norm2) + 1e-5);
}


