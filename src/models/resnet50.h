#ifndef _FACE_MOBILEFACENET_H_
#define _FACE_MOBILEFACENET_H_

#include "../../include/ncnnRay.hpp"
#include <vector>


class FeatureExtractor {
public:
    FeatureExtractor();

    ~FeatureExtractor();

    FeatureExtractor(const std::string &model_path, const std::string &model_name, const ncnn::Option &opt);

    void normalize(std::vector<float> &arr);

    std::vector<float> ExtractFeature(Image &image, const std::string &in_name, const std::string &out_name);

    float getSimilarity(const std::vector<float> &v1, const std::vector<float> &v2);

    float calculateSimilarity(const std::vector<float> &feat1, const std::vector<float> &feat2);

private:
    ncnn::Net net;
};


#endif // !_FACE_MOBILEFACENET_H_

