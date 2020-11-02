//#include "../include/ncnnRay.hpp" // MUST BE INCLUDED FIRST
//#include "raylib.h"
//#pragma warning( push, 0 )
//#pragma warning( disable : 4576 )
#define RAYGUI_IMPLEMENTATION
#define RAYGUI_SUPPORT_RICONS

#include "../include/raygui/raygui.h"
#include "../3rdparty/raylib-3.0.0/src/external/glad.h"
#include "../include/raygui/ricons.h"

#define PL_MPEG_IMPLEMENTATION

#include "../include/pl_mpeg/pl_mpeg.h"
#include "models/neural.h"
#include "models/LFFD.h"
#include "models/FaceDetector.h"

Font gamefont;

void handleDroppedFiles(const int screenWidth, const int screenHeight, int &pixelFormatActive, Image &image,
                        Texture2D &texture, bool &imageLoaded, float &imageScale);

void handleExport(char *fileName, const Image &image, bool imageLoaded, bool btnExport, bool &windowBoxActive,
                  Texture2D &texture);

void handleImageScaling(const int screenWidth, const int screenHeight, const Image &image, bool imageLoaded,
                        float &imageScale, Rectangle &imageRec);

int main() {

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
    std::string model_name = "candy";
    std::string model_name2 = "mosaic";

    NeuralStyle nstyle(model_path, model_name, opt);
    NeuralStyle nstyle2(model_path, model_name2, opt);

    LFFD lffd1(model_path, 8, 0, opt);

    Detector detector (model_path, opt, false);

    const int screenWidth = 1600;
    const int screenHeight = 1200;

    const char *mainTitle = "ncnnRay: Model Studio";
    InitWindow(screenWidth, screenHeight, mainTitle);
    gamefont = LoadFont("GameCube.ttf");
    Sound clickSound = LoadSound("save.ogg");
    Sound saveImageSound = LoadSound("click.ogg");

    const char *torchStyle = "torch2.rgs";
    GuiLoadStyle(torchStyle);
    GuiFade(0.9f);
    GuiSetStyle(DEFAULT, TEXT_SPACING, 1);
    Color defTextCLR = GetColor(GuiGetStyle(DEFAULT, LINE_COLOR));

    GuiPanel(Rectangle{0, 0, (float) GetScreenWidth(), (float) GetScreenHeight()});
    Rectangle windowBoxRec = {screenWidth / 2 - 110, screenHeight / 2 - 100, 220, 190};
    bool windowBoxActive = false;
    int fileFormatActive = 0;
    const char *fileFormatTextList[1] = {".png"};
    int pixelFormatActive = 0;
    const char *pixelFormatTextList[1] = {"RGB"};
    bool textBoxEditMode = false;
    char fileName[64] = "img0";
    //--------------------------------------------------------------------------------------
    struct videoState {
//        Image image = {0};
//        bool imageLoaded = false;
        bool videoLoaded = false;
        std::string videoFileName;
        plm_t *plm = NULL;
        int w;
        int h;
        double num_pixels = 0.0;
        long currentFrame = 1;
        Texture2D tx = {0};
        Image imageFrame = {0};
        plm_frame_t *frame = NULL;
        double dur = -1.0;
    };

    videoState vs = {0};

    float imageScale = 1.0f;
    float randomSeed = 0;
    Rectangle imageRec = {0.0f};
    bool btnExport = false;
    bool animate = false;
    bool isVulkan = true;
    int comboBoxActive = 0;
    SetTargetFPS(60);
    InitAudioDevice();
    // Default sizes.
    float buttonWidth = 120;
    float buttonHeight = 40;
    float padding = 50;
    float smallPadding = 40;
    float leftPadding = 160;

    auto statusBarRect = Rectangle{0, (float)GetScreenHeight() - 28, (float)GetScreenWidth(), 28};
    GuiPanel(Rectangle{0, 0, (float) GetScreenWidth(), (float) GetScreenHeight()});

    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        const char* statusText = TextFormat("GPU: %s", glGetString(GL_RENDERER)); //
        GuiStatusBar(statusBarRect, statusText);
//        handleExport(fileName, image, imageLoaded, btnExport, windowBoxActive, texture);
        handleImageScaling(screenWidth, screenHeight, vs.imageFrame, vs.videoLoaded, imageScale, imageRec);

        BeginDrawing();
        ClearBackground(DARKGRAY);
        DrawFPS(10, 10);
        DrawTextEx(GuiGetFont(), mainTitle, Vector2{10, 40}, 19, 1.0f, WHITE);

        if (IsFileDropped()) {
            int fileCount = 0;
            char **droppedFiles = GetDroppedFiles(&fileCount);
            if (fileCount == 1) {
                if (IsFileExtension(droppedFiles[0], ".mpg")) {
//
//                    if (vs.plm!=nullptr) {
//                        plm_destroy(vs.plm); // Free the old video
//                    }
//                    vs={0};// Throw the old one if it exists
                    vs.videoFileName = droppedFiles[0];
                    TraceLog(LOG_DEBUG, "ncnnRay: video:%s", vs.videoFileName.c_str());
                    vs.plm = plm_create_with_filename(vs.videoFileName.c_str());
                    if (!vs.plm) {
                        std::cout << "Couldn't open file:" << vs.videoFileName << std::endl;
                    } else {
                        TraceLog(LOG_DEBUG, "ncnnRay: video:%s opened - frame:%d", vs.videoFileName.c_str(),
                                 plm_get_duration(vs.plm));
                    }
                    plm_set_audio_enabled(vs.plm, FALSE);
                    vs.dur = plm_get_duration(vs.plm);
                    vs.w = plm_get_width(vs.plm);
                    vs.h = plm_get_height(vs.plm);
                    vs.num_pixels = vs.w * vs.h;
                    vs.videoLoaded = true;
                }
            }
            ClearDroppedFiles();
        }

        if (vs.tx.id > 0) {
            DrawTextureEx(vs.tx, Vector2{screenWidth / 2 - (float) vs.tx.width * imageScale / 2,
                                         screenHeight / 2 - (float) vs.tx.height * imageScale / 2}, 0.0f,
                          imageScale, WHITE);
            DrawRectangleLinesEx(imageRec, 1,
                                 CheckCollisionPointRec(GetMousePosition(), imageRec) ? RED : DARKGRAY);
            DrawText(FormatText("SCALE: %.2f%%", imageScale * 100.0f), 10, screenHeight - 60, 20, defTextCLR);

        } else {
            DrawTextEx(GuiGetFont(), "NCNN MODEL STUDIO: DRAG & DROP A VIDEO.",
                       Vector2{10, (float) GetScreenHeight() - 52},
                       19, 0.0f, WHITE);
            GuiDisable();
        }

        if (vs.videoLoaded) {
            if (vs.plm != nullptr) {
                vs.frame = plm_decode_video(vs.plm);
                if (vs.frame != nullptr) {
                    TraceLog(LOG_DEBUG, "ncnnRay: decoding- frame:%d", vs.frame->width);
                    if (vs.tx.width > 0) {
                        UnloadTexture(vs.tx);
                    }
                    if (vs.imageFrame.width > 0) {
                        UnloadImage(vs.imageFrame);
                    }
                    uint8_t *rgb_data = (uint8_t *) malloc(vs.num_pixels * 3);
                    plm_frame_to_rgb(vs.frame, rgb_data, vs.w * 3);
                    vs.imageFrame.data = rgb_data;
                    vs.imageFrame.width = vs.w;
                    vs.imageFrame.height = vs.h;
                    vs.imageFrame.format = UNCOMPRESSED_R8G8B8;
                    vs.imageFrame.mipmaps = 1;

                    vs.tx = LoadTextureFromImage(vs.imageFrame);
                    vs.currentFrame++;
                }
            } else {
                TraceLog(LOG_ERROR, "ncnnRay: Something fishy is going on ...");
                animate = false;
                GuiDisable();
//                vs.videoLoaded= false;
            }
        }

        if (!animate) {
            comboBoxActive = false;
        }

        comboBoxActive = GuiComboBox(
                Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 9 * padding, buttonWidth + 35,
                          buttonHeight},
                "RETINA;LFFD;CANDY;MOSAIC", comboBoxActive);
        animate = (GuiCheckBox(Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 8 * padding, 20, 20},
                               "Animate", animate));

        isVulkan = (GuiCheckBox(Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 11 * padding, 30, 20},
                               "Vulkan", isVulkan));

        if (animate && vs.videoLoaded) {
            TraceLog(LOG_DEBUG, "TorchRaLib: Animate");
//            ImageResize(&vs.imageFrame, vs.imageFrame.width/2, vs.imageFrame.height/2);
            if (comboBoxActive + 1 == 1) {
//                VU.applyModelOnImage(device, moduleMosaic, vs.imageFrame);
                detector.detectFaces(vs.imageFrame);
//                TraceLog(LOG_INFO, "ncnnRay: RetinaFace");
                vs.tx = LoadTextureFromImage(vs.imageFrame);
                DrawTextureEx(vs.tx, Vector2{screenWidth / 2 - (float) vs.tx.width * imageScale / 2,
                                             screenHeight / 2 - (float) vs.tx.height * imageScale / 2}, 0.0f,
                              imageScale, WHITE);
//                UnloadTexture(vs.tx);
            }

            if (comboBoxActive + 1 == 2) {
//                VU.applyModelOnImage(device, moduleMosaic, vs.imageFrame);
                lffd1.detectFacesAndDrawOnImage(vs.imageFrame);
                vs.tx = LoadTextureFromImage(vs.imageFrame);
                DrawTextureEx(vs.tx, Vector2{screenWidth / 2 - (float) vs.tx.width * imageScale / 2,
                                             screenHeight / 2 - (float) vs.tx.height * imageScale / 2}, 0.0f,
                              imageScale, WHITE);
//                UnloadTexture(vs.tx);
            }

            if (comboBoxActive + 1 == 3) {
                vs.imageFrame = nstyle.applyStyleOnImage(vs.imageFrame);
                vs.tx = LoadTextureFromImage(vs.imageFrame);
                DrawTextureEx(vs.tx, Vector2{screenWidth / 2 - (float) vs.tx.width * imageScale / 2,
                                             screenHeight / 2 - (float) vs.tx.height * imageScale / 2}, 0.0f,
                              imageScale, WHITE);
            }

            if (comboBoxActive + 1 == 4) {
                vs.imageFrame = nstyle2.applyStyleOnImage(vs.imageFrame);
                vs.tx = LoadTextureFromImage(vs.imageFrame);
                DrawTextureEx(vs.tx, Vector2{screenWidth / 2 - (float) vs.tx.width * imageScale / 2,
                                             screenHeight / 2 - (float) vs.tx.height * imageScale / 2}, 0.0f,
                              imageScale, WHITE);
            }

//            ImageColorContrast(&vs.imageFrame, 10.0);
            vs.tx = LoadTextureFromImage(vs.imageFrame);
            DrawTextureEx(vs.tx, Vector2{screenWidth / 2 - (float) vs.tx.width * imageScale / 2,
                                         screenHeight / 2 - (float) vs.tx.height * imageScale / 2}, 0.0f, imageScale,
                          WHITE);

            statusText = TextFormat("Model# %i", comboBoxActive + 1);
            GuiStatusBar(statusBarRect, statusText);
        }

        GuiEnable();
        EndDrawing();
    }

    UnloadImage(vs.imageFrame);
    UnloadTexture(vs.tx);
    CloseWindow();        // Close window and OpenGL context
    return 0;
}

void handleImageScaling(const int screenWidth, const int screenHeight, const Image &image, bool imageLoaded,
                        float &imageScale, Rectangle &imageRec) {
    if (imageLoaded) {
        imageScale += (float) GetMouseWheelMove() * 0.15f;   // Image scale control
        if (imageScale <= 0.1f) imageScale = 0.1f;
        else if (imageScale >= 4) imageScale = 4;

        imageRec = Rectangle{screenWidth / 2 - (float) image.width * imageScale / 2,
                             screenHeight / 2 - (float) image.height * imageScale / 2,
                             (float) image.width * imageScale, (float) image.height * imageScale};
    }
}

