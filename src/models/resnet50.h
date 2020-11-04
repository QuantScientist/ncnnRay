#ifndef _FACE_MOBILEFACENET_H_
#define _FACE_MOBILEFACENET_H_

#include "../../include/ncnnRay.hpp"
#include <vector>


class FeatureExtractor{
public:
    FeatureExtractor();
	~FeatureExtractor();
    FeatureExtractor(const std::string &model_path,const ncnn::Option &opt);

//	int ExtractFeature(const Image& img_face, std::vector<float>* feature);
    void normalize(std::vector<float>& arr);
    int ExtractFeature(Image  &image,std::vector<float>* feature);
    double getSimilarity(const std::vector<float>& v1, const std::vector<float>& v2);

private:
	ncnn::Net net;
    int kFeatureDim=2048;
    std::string prefix="mobilefacenets"; //resnet50-opt
};


#endif // !_FACE_MOBILEFACENET_H_

