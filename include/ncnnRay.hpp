#pragma once
#define NOGDICAPMASKS // CC_, LC_, PC_, CP_, TC_, RC_
#define NOVIRTUALKEYCODES // VK_*
#define NOWINMESSAGES // WM_, EM_, LB_, CB_
#define NOWINSTYLES // WS_, CS_, ES_, LBS_, SBS_, CBS_
#define NOSYSMETRICS // SM_*
#define NOMENUS // MF_*
#define NOICONS // IDI_*
#define NOKEYSTATES // MK_*
#define NOSYSCOMMANDS // SC_*
#define NORASTEROPS // Binary and Tertiary raster ops
#define NOSHOWWINDOW // SW_*
#define OEMRESOURCE // OEM Resource values
#define NOATOM // Atom Manager routines
#define NOCLIPBOARD // Clipboard routines
#define NOCOLOR // Screen colors
#define NOCTLMGR // Control and Dialog routines
#define NODRAWTEXT // DrawText() and DT_*
#define NOGDI // All GDI defines and routines
#define NOKERNEL // All KERNEL defines and routines
#define NOUSER // All USER defines and routines
#define NONLS // All NLS defines and routines/
#define NOMB // MB_* and MessageBox()
#define NOMEMMGR // GMEM_, LMEM_, GHND, LHND, associated routines
#define NOMETAFILE // typedef METAFILEPICT
//#define NOMINMAX // Macros min(a,b) and max(a,b)
#define NOMSG // typedef MSG and associated routines
#define NOOPENFILE // OpenFile(), OemToAnsi, AnsiToOem, and OF_*
#define NOSCROLL // SB_* and scrolling routines
#define NOSERVICE // All Service Controller routines, SERVICE_ equates, etc.
#define NOSOUND // Sound driver routines
#define NOTEXTMETRIC // typedef TEXTMETRIC and associated routines
#define NOWH // SetWindowsHook and WH_*
#define NOWINOFFSETS // GWL_, GCL_, associated routines
#define NOCOMM // COMM driver routines
#define NOKANJI // Kanji support stuff.
#define NOHELP // Help engine interface.
#define NOPROFILER // Profiler interface.
#define NODEFERWINDOWPOS // DeferWindowPos routines
#define NOMCX // Modem Configuration Extensions
#pragma warning( push, 0 )
#pragma warning( disable : 4576 )

#include "raylib.h"

//#define NOUSER //Resolve ray error  error C2660: 'LoadImageA': function does not take 1 arguments
//#include <intrin.h> //Resolve ray error  ppltasks.h(2712): error C3861: '_ReturnAddress': identifier not found
#include <iostream>
#include <iostream>
#include <chrono>
#include <iostream>
#include <typeinfo>
#include <thread>
#include <future>

//############################################ ncnn ##########################################
#include <float.h>
#include <stdio.h>

#ifdef _WIN32

#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif
//#include "benchmark.h"
#include "cpu.h"
#include "net.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"
#include "benchmark.h"
#include <cstring> //to import the std::memcpy function.
#include <iostream>
#include <string.h>
//#include "../include/easylogging/easylogging++.h"
//INITIALIZE_EASYLOGGINGPP;

#include <cmath>
#include <string>
#include <chrono>
#include <algorithm>
#include <iostream>

#ifndef SCOPETIMER__3472893
#define SCOPETIMER__3472893

#include <iostream>
#include <chrono>

struct ScopeTimer
{
    std::chrono::high_resolution_clock::time_point start;
    const char* title;

    ScopeTimer()
    {
        title = nullptr;
        start = std::chrono::high_resolution_clock::now();
    }

    ScopeTimer(const char* title)
    {
        this->title = title;
        start = std::chrono::high_resolution_clock::now();
    }

    ~ScopeTimer()
    {
        using std::chrono::duration_cast;
        using std::chrono::microseconds;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start)/1000.0;
        std::cout << (title ? title : "") << ": " << duration.count() << " ms\n";
    }
};

#endif

#if EMSCRIPTEN
#include <emscripten.h>
#include <fetch.h>
#include <emscripten/fetch.h>
#include <emscripten/emscripten.h>

static void mydownloadFailed(emscripten_fetch_t *fetch) {
    TraceLog(LOG_INFO,"Downloading %s failed, HTTP failure status code: %d.\n", fetch->url, fetch->status);
    emscripten_fetch_close(fetch); // Also free data on failure.
}
#endif

//############################################ ncnn ##########################################

static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
static ncnn::VulkanDevice *g_vkdev = 0;
static ncnn::VkAllocator *g_blob_vkallocator = 0;
static ncnn::VkAllocator *g_staging_vkallocator = 0;
#endif // NCNN_VULKAN
using namespace std;
using namespace std::chrono;

static int tensorDIMS(const ncnn::Mat &tensor) {
    return tensor.dims;
}

static ncnn::Option optGPU(bool use_vulkan_compute = false, int gpu_device = -1) {
    ncnn::Option opt;
    opt.lightmode = true;
//    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_vulkan_compute = use_vulkan_compute;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = false;

//    opt.use_winograd_convolution = true;
//    opt.use_sgemm_convolution = true;
//    opt.use_int8_inference = true;
//    opt.use_fp16_packed = true;
//    opt.use_fp16_storage = true;
//    opt.use_fp16_arithmetic = true;
//    opt.use_int8_storage = true;
//    opt.use_int8_arithmetic = true;
//    opt.use_packing_layout = true;
//    opt.use_shader_pack8 = false;
//    opt.use_image_storage = false;

    opt.use_vulkan_compute = use_vulkan_compute;
    #if NCNN_VULKAN
    if (use_vulkan_compute && gpu_device > -1) {
        TraceLog(LOG_INFO, "ncnnRay: use_vulkan_compute:%i", use_vulkan_compute);
//        ncnn::create_gpu_instance();
        opt.use_vulkan_compute = true;

        opt.blob_vkallocator = g_blob_vkallocator;
        opt.workspace_vkallocator = g_blob_vkallocator;
        opt.staging_vkallocator = g_staging_vkallocator;
    }
    #endif // NCNN_VULKAN

    return opt;
}

static void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

static int isGPU() {
    // initialize when app starts
    int ins=0;
    #if NCNN_VULKAN
    ins = ncnn::get_gpu_count();
    #endif // NCNN_VULKAN
    std::cout << "GPU instance=?:" << ins << std::endl;;
//    auto g= ncnn::get_gpu_device(0);
//    std::cout<<"GPU Device=?:" << g <<std::endl;;
    return ins;
}

static ncnn::Mat rayImageToNcnn(const Image &image) {
    size_t width = image.width;
    size_t height = image.height;
    int dataSize = GetPixelDataSize(width, height, image.format);
    TraceLog(LOG_INFO, "ncnnRay: total image pixels:%i", dataSize);

    int bytesPerPixel = dataSize / (width * height);
    TraceLog(LOG_INFO, "ncnnRay: image bytesPerPixel:%i", bytesPerPixel);
//    if (bytesPerPixel==4) bytesPerPixel=3;

    auto pointer = new unsigned char[dataSize];
    const unsigned char *imagePointer = (unsigned char *) image.data;
    std::memcpy(pointer, imagePointer, dataSize);
//    error C2664: 'ncnn::Mat ncnn::Mat::from_pixels(const unsigned char *,int,int,int,ncnn::Allocator *)': cannot convert argument 1 from 'void *const ' to 'const unsigned char *'
//    ncnn::Mat tensor = ncnn::Mat::from_pixels(static_cast<const unsigned char *>(image.data), ncnn::Mat::PIXEL_BGR2RGB, width, height);
    int type = ncnn::Mat::PIXEL_RGB;
    if (bytesPerPixel == 4) {
        type = ncnn::Mat::PIXEL_RGBA2RGB;
    }
//    PIXEL_RGB = 1,
//    PIXEL_BGR = 2,
//    PIXEL_GRAY = 3,
//    PIXEL_RGBA = 4,
//    PIXEL_BGRA = 5,

//    TraceLog(LOG_INFO, "ncnnRay: type:%i", type);
//    ncnn::Mat tensor = ncnn::Mat::from_pixels(static_cast<const unsigned char *>(image.data), ncnn::Mat::PIXEL_RGB, width, height);
    ncnn::Mat tensor = ncnn::Mat::from_pixels((const unsigned char *) image.data, type, width, height);

    TraceLog(LOG_INFO, "ncnnRay: rayImageToNcnn final T dims:%i", tensor.shape().dims);
//    delete[] pointer;
    return tensor;
}

static Image ncnnToRayImage(ncnn::Mat &tensor) {
    size_t width = tensor.w;
    size_t height = tensor.h;
    unsigned char *torchPointer = reinterpret_cast<unsigned char *>(RL_MALLOC(
            3 * height * width * sizeof(unsigned char)));
    tensor.to_pixels(torchPointer, ncnn::Mat::PIXEL_RGB);
    TraceLog(LOG_INFO, "ncnnRay: ncnnToRayImage final T dims:%i", tensor.shape().dims);
//    return Image{0};
    return Image{
            torchPointer,
            (int) width,
            (int) height,
            1, //that line is mipmaps, keep as 1
            UNCOMPRESSED_R8G8B8}; //its an enum specifying formar, 8 bit R, 8 bit G, 8 bit B, no alpha UNCOMPRESSED_R8G8B8A8 UNCOMPRESSED_R8G8B8
}
