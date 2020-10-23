//#include "../include/ncnnRay.hpp" // MUST BE INCLUDED FIRST
//#include "raylib.h"
//#pragma warning( push, 0 )
//#pragma warning( disable : 4576 )
#define RAYGUI_IMPLEMENTATION
#define RAYGUI_SUPPORT_RICONS

#include "../include/raygui/raygui.h"
//#include "../bin64/_deps/raylib-src/src/external/glad.h"
#include "../include/raygui/ricons.h"

#define PL_MPEG_IMPLEMENTATION

#include "../include/pl_mpeg/pl_mpeg.h"
#include "models/neural.h"
#include "models/LFFD.h"
#include "models/FaceDetector.h"

#if EMSCRIPTEN
#include <emscripten.h>
//#include "../web/_deps/raylib-src/src/external/glad.h"
#endif

#include "models/FaceDetector.h"

//----------------------------------------------------------------------------------
// Global Variables Definition
//----------------------------------------------------------------------------------
int screenWidth = 1200;
int screenHeight = 800;

//----------------------------------------------------------------------------------
// Module Functions Declaration
//----------------------------------------------------------------------------------
void UpdateDrawFrame(void);     // Update and Draw one frame

void handleImageScaling(const int screenWidth, const int screenHeight, const Image &image, bool imageLoaded,
                        float &imageScale, Rectangle &imageRec);

void handleDroppedFiles(const int screenWidth, const int screenHeight, Image &image,
                        Texture2D &texture, bool &imageLoaded, float &imageScale);

void handleExport(char *fileName, const Image &image, bool imageLoaded, bool btnExport, bool &windowBoxActive,
                  Texture2D &texture);


//----------------------------------------------------------------------------------
// Main Enry Point
//----------------------------------------------------------------------------------
Image image = {0};
Texture2D texture = {0};
bool imageLoaded = false;
bool vidLoaded = false;
float imageScale = 1.0f;
float randomSeed = 0;
Rectangle imageRec = {0.0f};
const char *mainTitle = "ncnnRay: WebAssembly AI studio";
Color defTextCLR = GetColor(GuiGetStyle(DEFAULT, LINE_COLOR));


// Default sizes.
float buttonWidth = 120;
float buttonHeight = 40;
float padding = 50;
float smallPadding = 40;
float leftPadding = 160;

auto statusBarRect = Rectangle{0, (float) GetScreenHeight() - 28, (float) GetScreenWidth(), 28};


//std::unique_ptr<Detector> _det;

// Why am I using pointers? WASM loop which requires a method without parameters
Detector *detector = nullptr;
NeuralStyle *nstyle1 = nullptr;
NeuralStyle *nstyle2 = nullptr;
LFFD *lffd = nullptr;

bool windowBoxActive = false;
bool btnExport = false;
char fileName[64] = "img0";
Rectangle windowBoxRec = {};
int fileFormatActive = 0;
const char *fileFormatTextList[1] = {".png"};
int pixelFormatActive = 0;
const char *pixelFormatTextList[1] = {"RGB"};
bool textBoxEditMode = false;

bool badFileExt=false;

int main() {
    bool use_vulkan_compute = true;
#if EMSCRIPTEN
    use_vulkan_compute = false;
#endif

#if (PLATFORM_WEB)
    use_vulkan_compute = false;
#endif
    int gpu_device = 0;

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

    std::string model_path = ".";
    ncnn::Option opt = optGPU(use_vulkan_compute, gpu_device);
    TraceLog(LOG_INFO, "ncnnRay: Opt using vulkan::%i", opt.use_vulkan_compute);
//    LFFD lffd1(model_path, 8, 0, opt);
    std::string model_name = "candy";
    std::string model_name2 = "mosaic";

    windowBoxRec = {static_cast<float>(screenWidth / 2 - 110), static_cast<float>(screenHeight / 2 - 100), 220, 190};

    detector = new Detector(model_path, opt, false);
    nstyle1 = new NeuralStyle(model_path, model_name, opt);
    nstyle2 = new NeuralStyle(model_path, model_name2, opt);

    lffd = new LFFD(model_path, 8, 0, opt);

    TraceLog(LOG_INFO, "ncnnRay: models using vulkan::%i", detector->Net->opt.use_vulkan_compute);

    #if EMSCRIPTEN
    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    strcpy(attr.requestMethod, "GET");
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.onsuccess = downloadSucceeded;
    attr.onerror = downloadFailed;
    //emscripten_fetch(&attr, "http://localhost:9999/update.sh");

//    emscripten_fetch(&attr, "https://raw.githubusercontent.com/juj/emscripten/emscripten_fetch/system/include/emscripten/fetch.h");
//    emscripten_fetch(&attr, "https://pbs.twimg.com/media/CAoh21KWEAARhYx.png");
//    emscripten_fetch(&attr, "http://localhost:8000/web/");
    #endif // NCNN_VULKAN

    // Initialization
    //--------------------------------------------------------------------------------------
    InitWindow(screenWidth, screenHeight, mainTitle);
    const char *torchStyle = "torch2.rgs";
    GuiLoadStyle(torchStyle);
    GuiFade(0.9f);
    GuiSetStyle(DEFAULT, TEXT_SPACING, 1);
    // TTF Font loading with custom generation parameters
    Font font = LoadFontEx("GameCube.ttf", 18, 0, 0);
    GuiSetFont(font);
//    std::string fileName = "faces.png";
//    image = LoadImage(fileName.c_str());
//    texture = LoadTextureFromImage(image);

//    GuiPanel(Rectangle{0, 0, (float) GetScreenWidth(), (float) GetScreenHeight()});

#if (PLATFORM_WEB)
    emscripten_set_main_loop(UpdateDrawFrame, 0, 1);
#else
    SetTargetFPS(60);   // Set our game to run at 60 frames-per-second
    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        UpdateDrawFrame();
    }
#endif
    CloseWindow();        // Close window and OpenGL context
    return 0;
}

//----------------------------------------------------------------------------------
// Module Functions Definition
//----------------------------------------------------------------------------------
void UpdateDrawFrame(void) {
    handleDroppedFiles(screenWidth, screenHeight, image, texture, imageLoaded, imageScale);
    handleExport(fileName, image, imageLoaded, btnExport, windowBoxActive, texture);
    handleImageScaling(screenWidth, screenHeight, image, imageLoaded, imageScale, imageRec);
    handleImageScaling(screenWidth, screenHeight, image, imageLoaded, imageScale, imageRec);
    BeginDrawing();
    ClearBackground(DARKGRAY);
    DrawFPS(10, 10);
    DrawTextEx(GuiGetFont(), mainTitle, Vector2{10, 40}, 19, 1.0f, WHITE);

    if (texture.id > 0) {
        if (texture.id > 0) {
            DrawTextureEx(texture, Vector2{screenWidth / 2 - (float) texture.width * imageScale / 2,
                                           screenHeight / 2 - (float) texture.height * imageScale / 2}, 0.0f,
                          imageScale, WHITE);
            DrawRectangleLinesEx(imageRec, 1,
                                 CheckCollisionPointRec(GetMousePosition(), imageRec) ? RED : DARKGRAY);
            DrawText(FormatText("SCALE: %.2f%%", imageScale * 100.0f), 10, screenHeight - 60, 20, defTextCLR);
        }

    } else {
        DrawTextEx(GuiGetFont(), "Drag and drop a PNG image onto the canvas.\n"
                                 "Use your scroll wheel to zoom in and out.",
                   Vector2{10, (float) GetScreenHeight() - 52},
                   19, 0.0f, WHITE);
//        badFileExt=false;
        GuiDisable();
    }

    if (GuiButton(
            Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 2 * padding, buttonWidth, buttonHeight},
            "LFFD")) {
        lffd->detectFacesAndDrawOnImage(image);
        TraceLog(LOG_INFO, "ncnnRay: retina");
        UnloadTexture(texture);
        texture = LoadTextureFromImage(image);
    }

    if (GuiButton(
            Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 3 * padding, buttonWidth, buttonHeight},
            "Mosaic")) {
//        ImageResize(&image,image.width/2, image.height/2);
        image = nstyle2->applyStyleOnImage(image);
//        ImageResize(&image,image.width*4, image.height*4);
        TraceLog(LOG_INFO, "ncnnRay: Mosaic");
        UnloadTexture(texture);
        texture = LoadTextureFromImage(image);
    }

    if (GuiButton(
            Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 4 * padding, buttonWidth, buttonHeight},
            "RetinaF")) {
        detector->detectFaces(image);
        TraceLog(LOG_INFO, "ncnnRay: retina");
        UnloadTexture(texture);
        texture = LoadTextureFromImage(image);
    }


    if (GuiButton(
            Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 5 * padding, buttonWidth, buttonHeight},
            "Candy")) {
//        ImageResize(&image, image.width / 4, image.height / 4);
        image = nstyle1->applyStyleOnImage(image);
        TraceLog(LOG_INFO, "ncnnRay: candy");
//        ImageColorBrightness(&image, -40);
        texture = LoadTextureFromImage(image);
    }

    if (GuiButton(
            Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 6 * padding, buttonWidth, buttonHeight},
            "Lighter")) {
//            PlaySound(clickSound);
        TraceLog(LOG_INFO, "ncnnRay: Lighter");
        ImageColorBrightness(&image, +40);
        texture = LoadTextureFromImage(image);
    }

    if (GuiButton(Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 10 * padding, buttonWidth, buttonHeight},"SAVE")) {
//        PlaySound(saveImageSound);
        TraceLog(LOG_INFO, "ncnnRay: save image");
        windowBoxActive = true;
    }


    if (windowBoxActive) {
        DrawRectangle(0, 0, screenWidth, screenHeight,
                      Fade(GetColor(GuiGetStyle(DEFAULT, BACKGROUND_COLOR)), 0.75f));
        windowBoxActive = !GuiWindowBox(Rectangle{windowBoxRec.x, windowBoxRec.y, 290, 220},
                                        "Save Options");
        GuiLabel(Rectangle{windowBoxRec.x + 10, windowBoxRec.y + 35, 60, 25}, "Format:");
        fileFormatActive = GuiComboBox(Rectangle{windowBoxRec.x + 80, windowBoxRec.y + 35, 130, 25},
                                       TextJoin(fileFormatTextList, 1, ";"), fileFormatActive);
        GuiLabel(Rectangle{windowBoxRec.x + 10, windowBoxRec.y + 70, 63, 25}, "Pixel:");
        pixelFormatActive = GuiComboBox(Rectangle{windowBoxRec.x + 80, windowBoxRec.y + 70, 130, 25},
                                        TextJoin(pixelFormatTextList, 1, ";"), pixelFormatActive);
        GuiLabel(Rectangle{windowBoxRec.x + 10, windowBoxRec.y + 105, 50, 25}, "Name:");
        if (GuiTextBox(Rectangle{windowBoxRec.x + 80, windowBoxRec.y + 105, 130, 25},
                       fileName, 64, textBoxEditMode))
            textBoxEditMode = !textBoxEditMode;

        btnExport = GuiButton(Rectangle{windowBoxRec.x + 10, windowBoxRec.y + 145, 260, 40}, "Save Image");
    } else btnExport = false;

    if (btnExport) DrawText("Image saved", 20, screenHeight - 20, 20, RED);

    if (badFileExt) {
        GuiEnable();
        DrawRectangle(0, 0, screenWidth, screenHeight,
                      Fade(GetColor(GuiGetStyle(DEFAULT, BACKGROUND_COLOR)), 0.75f));
        GuiWindowBox(Rectangle{windowBoxRec.x, windowBoxRec.y, 330, 100},
                                        "Warning.");
//        badFileExt=false;
        GuiLabel(Rectangle{windowBoxRec.x + 10, windowBoxRec.y + 35, 300, 25}, "Only PNG files are supported.");
//        fileFormatActive = GuiComboBox(Rectangle{windowBoxRec.x + 80, windowBoxRec.y + 35, 130, 25},
//                                       TextJoin(fileFormatTextList, 1, ";"), fileFormatActive);
//        GuiLabel(Rectangle{windowBoxRec.x + 10, windowBoxRec.y + 70, 63, 25}, "Pixel:");
//        pixelFormatActive = GuiComboBox(Rectangle{windowBoxRec.x + 80, windowBoxRec.y + 70, 130, 25},
//                                        TextJoin(pixelFormatTextList, 1, ";"), pixelFormatActive);
//        GuiLabel(Rectangle{windowBoxRec.x + 10, windowBoxRec.y + 105, 50, 25}, "Name:");
//        if (GuiTextBox(Rectangle{windowBoxRec.x + 80, windowBoxRec.y + 105, 130, 25},
//                       fileName, 64, textBoxEditMode))
//            textBoxEditMode = !textBoxEditMode;

        badFileExt = !GuiButton(Rectangle{windowBoxRec.x + 10, windowBoxRec.y + 65, 300, 25}, "OK");
    }

    GuiEnable();
    EndDrawing();
    //----------------------------------------------------------------------------------
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

void handleDroppedFiles(const int screenWidth, const int screenHeight, Image &image,
                        Texture2D &texture, bool &imageLoaded, float &imageScale) {
    if (IsFileDropped()) {
        int fileCount = 0;
        char **droppedFiles = GetDroppedFiles(&fileCount);

        if (fileCount == 1) {
            if (IsFileExtension(droppedFiles[0], ".png")) {
                badFileExt=false;
                TraceLog(LOG_INFO, "ncnnRay: image");
                Image imTemp = LoadImage(droppedFiles[0]);
                if (imTemp.data != nullptr) {
                    UnloadImage(image);
                    image = imTemp;
                    UnloadTexture(texture);
                    texture = LoadTextureFromImage(image);
                    imageLoaded = true;
                    if (texture.height > texture.width)
                        imageScale = (float) (screenHeight - 100) / (float) texture.height;
                    else imageScale = (float) (screenWidth - 100) / (float) texture.width;
                }
            }
            else{
                badFileExt=true;
            }
        }

        ClearDroppedFiles();
    }
}

void handleExport(char *fileName, const Image &image, bool imageLoaded, bool btnExport, bool &windowBoxActive,
                  Texture2D &texture) {
    if (btnExport) {
        if (imageLoaded) {
//                ImageFormat(&image,UNCOMPRESSED_R8G8B8A8);
            if ((GetExtension(fileName) == nullptr) || (!IsFileExtension(fileName, ".png")))
                strcat(fileName, ".png\0");     // No extension provided
            ExportImage(image, fileName);
            UnloadTexture(texture);
            texture = LoadTextureFromImage(image);
        }
        windowBoxActive = false;
    }
}