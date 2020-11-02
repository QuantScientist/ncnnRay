#include "neural.h"

NeuralStyle::NeuralStyle() {};

NeuralStyle::NeuralStyle(const std::string &model_path,
                         const std::string &model_name,
                         const ncnn::Option &opt
) {

    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute) {
        TraceLog(LOG_INFO, "ncnnRay: Opt using vulkan::%i", net.opt.use_vulkan_compute);
        net.set_vulkan_device(g_vkdev);
    }
    #endif // NCNN_VULKAN

    std::string bin = model_path + "/" + model_name + ".bin";
    net.load_param(styletransfer_param_bin);
    net.load_model(bin.data());
    TraceLog(LOG_INFO, "ncnnRay: model loaded, GPU enabled?=%i", net.opt.use_vulkan_compute);

}

NeuralStyle::~NeuralStyle() = default;

Image NeuralStyle::applyStyleOnImage(Image &image) {
    ScopeTimer Tmr("NeuralStyle::applyStyleOnImage");
    ncnn::Mat inmat = rayImageToNcnn(image);
    cout << "Total in:" << inmat.total() << endl;
    cout << "Dims in:" << tensorDIMS(inmat) << endl;;
//    lffd0.detect(inmat, face_info, 240, 320);
    ncnn::Mat out = transform(inmat);
//    lffd.detect(inmat, face_info, 240, 320);
    cout << "Total out:" << out.total() << endl;
    cout << "Dims out:" << tensorDIMS(out) << endl;;
    Image saveImage = ncnnToRayImage(out);
    return saveImage;
//    ImageFormat(&image,UNCOMPRESSED_R8G8B8A8);
//    Texture2D texture = LoadTextureFromImage(image);
}

ncnn::Mat NeuralStyle::transform(ncnn::Mat &in) {
    // styletransfer
//    std::cout<< "styletransfer ex using vulkan in :" <<in.dims <<std::endl;
    ncnn::Mat out;
    {
        ncnn::Extractor ex = net.create_extractor();
#if NCNN_VULKAN
//        if (net.opt.use_vulkan_compute)
//        {
//            net.set_vulkan_device(g_vkdev);
//            ex.set_vulkan_compute(true);
//            std::cout<< "ex using vulkan:" <<net.opt.use_vulkan_compute <<std::endl;
//        }
#endif // NCNN_VULKAN

        ex.input(styletransfer_param_id::BLOB_input1, in);
        ex.extract(styletransfer_param_id::BLOB_output1, out);
//        std::cout<< "styletransfer ex finished out:" <<out.dims <<std::endl;
    }

    return out;
}