#include "resnet50.h"
#include <iostream>
#include <string>

FeatureExtractor::FeatureExtractor() {}

FeatureExtractor::~FeatureExtractor() {
	net.clear();
}

FeatureExtractor::FeatureExtractor(const std::string &model_path,
                         const ncnn::Option &opt
) {
    net.opt = opt;
#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute) {
        TraceLog(LOG_INFO, "ncnnRay: FeatureExtractor Opt using vulkan::%i", net.opt.use_vulkan_compute);
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN
    std::string param = model_path + "/resnet50-opt.param";
    std::string bin = model_path + "/resnet50-opt.bin";
    net.load_param(param.c_str());
    net.load_model(bin.c_str());
    TraceLog(LOG_INFO, "ncnnRay: FeatureExtractor model loaded, GPU enabled?=%i", net.opt.use_vulkan_compute);

}

int FeatureExtractor::ExtractFeature(Image  &image,std::vector<float>* feature) {
    ScopeTimer Tmr("FeatureExtractor::ExtractFeature");
	feature->clear();
    ncnn::Mat in = rayImageToNcnn(image);

//    const float mean_vals[3] = { 0.f, 0.f, 0.f };
//    const float norm_vals[3] = { 1.0 / 255, 1.0 / 255, 1.0 / 255 };
//    in.substract_mean_normalize(mean_vals, norm_vals);

	ncnn::Extractor ex = net.create_extractor();
	ex.input("input.1", in);
	ncnn::Mat out;

	ex.extract("652", out); // Convolution Conv_116 1 1 488 652 0=2048 1=1 5=1
	TraceLog(LOG_INFO, "ncnnRay: fv2048 out.c =%d", out.c);
    std::vector<float> vec;
    for (int i = 0; i < out.c; ++i) {
        vec.emplace_back(*(static_cast<float*>(out.data) + i));
//        TraceLog(LOG_INFO, "ncnnRay: fv2048 =%d", *(static_cast<float*>(out.data) + i));
    }
    TraceLog(LOG_INFO, "ncnnRay: fv2048 vec =%d", vec.size());

    for (int i = 0; i < kFeatureDim; ++i) {
		feature->at(i) = out[i];
	}
    TraceLog(LOG_INFO, "ncnnRay: fv2048 vec-2 =%d", feature->size());


//	for (int i = 0; i < kFeatureDim; ++i) {
//		feature->at(i) = out[i];
//	}

//    TraceLog(LOG_INFO, "ncnnRay: fv2048=%d", feature);
	return 0;
}

void FeatureExtractor::normalize(std::vector<float>& arr) {
    double mod = 0.0;

    for (float i : arr) {
        mod += i * i;
    }

    double mag = std::sqrt(mod);

    if (mag == 0) {
        throw std::logic_error("The input vector is a zero vector");
    }

    for (float & i : arr) {
        i /= mag;
    }
}

double FeatureExtractor::getSimilarity(const std::vector<float>& v1, const std::vector<float>& v2) {
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


