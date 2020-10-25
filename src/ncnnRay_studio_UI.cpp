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

    NeuralStyle nstyle(model_path, model_name,  opt);
    NeuralStyle nstyle2(model_path, model_name2, opt);

    LFFD lffd1(model_path, 8, 0, opt);

    Detector detector (model_path, opt, false);
    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 1600;
    const int screenHeight = 1200;

    const char *mainTitle = "ncnnRay: Model Studio";
    InitWindow(screenWidth, screenHeight, mainTitle);
    gamefont = LoadFont("GameCube.ttf");
    Sound clickSound = LoadSound("save.ogg");
    Sound saveImageSound = LoadSound("click.ogg");

    //UI
    const char * torchStyle="torch2.rgs";
    GuiLoadStyle(torchStyle);
    GuiFade(0.9f);
    GuiSetStyle(DEFAULT, TEXT_SPACING, 1);
    Color defTextCLR = GetColor(GuiGetStyle(DEFAULT, LINE_COLOR));
//    SetTextureFilter(GetFontDefault().texture, FILTER_POINT); // Fix for HighDPI display problems
//    SetTextureFilter(font.texture, FILTER_BILINEAR)

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
    Image image = {0};
    Texture2D texture = {0};
    bool imageLoaded = false;
    bool vidLoaded = false;
    float imageScale = 1.0f;
    float randomSeed = 0;
    Rectangle imageRec = {0.0f};
    bool btnExport = false;
    bool animate = false;
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
//    while (!WindowShouldClose()) {
//        if (scene == 1) Splashscreen();
//        if (scene == 2) exit(0);
//    }

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
//       // Fullscreen
//        if (IsKeyReleased(KEY_F11)) {
//            ToggleFullscreen();
//        }
        const char* statusText = TextFormat("GPU: %s", glGetString(GL_RENDERER)); //
        GuiStatusBar(statusBarRect, statusText);
        //----------------------------------------------------------------------------------
        handleDroppedFiles(screenWidth, screenHeight, pixelFormatActive, image, texture, imageLoaded, imageScale);
        handleExport(fileName, image, imageLoaded, btnExport, windowBoxActive, texture);
        handleImageScaling(screenWidth, screenHeight, image, imageLoaded, imageScale, imageRec);
        BeginDrawing();
        ClearBackground(DARKGRAY);
        DrawFPS(10, 10);
        DrawTextEx(GuiGetFont(),mainTitle,Vector2 { 10, 40},19,1.0f,WHITE);

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
            DrawTextEx(GuiGetFont(),"ncnn MODEL STUDIO: DRAG & DROP AN IMAGE.",Vector2 { 10, (float)GetScreenHeight() - 52 },
                       19,0.0f,WHITE);
            GuiDisable();
        }
        

        if (GuiButton(Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 2 * padding, buttonWidth, buttonHeight}, "LffdF")) {
            lffd1.detectFacesAndDrawOnImage(image);
            TraceLog(LOG_INFO, "ncnnRay: LFFD");
            UnloadTexture(texture);
            texture = LoadTextureFromImage(image);
        }


        if (GuiButton(Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 3 * padding, buttonWidth, buttonHeight}, "Udiny")) {
            image = nstyle.applyStyleOnImage(image);
            TraceLog(LOG_INFO, "ncnnRay: Udi");

            UnloadTexture(texture);
            texture = LoadTextureFromImage(image);
        }

        if (GuiButton(Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 4 * padding, buttonWidth, buttonHeight}, "Mosaic")) {
            image = nstyle2.applyStyleOnImage(image);
            TraceLog(LOG_INFO, "ncnnRay: Mosaic");
            
            UnloadTexture(texture);
            texture = LoadTextureFromImage(image);
        }

        if (GuiButton(Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 5 * padding, buttonWidth, buttonHeight}, "RetinaF")) {
            detector.detectFaces(image);
            TraceLog(LOG_INFO, "ncnnRay: RetinaFace");
            ImageColorBrightness(&image, -40);
            texture = LoadTextureFromImage(image);
        }

        if (GuiButton(Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 6 * padding, buttonWidth, buttonHeight}, "Lighter")) {
//            PlaySound(clickSound);
            TraceLog(LOG_INFO, "ncnnRay: Lighter");
            ImageColorBrightness(&image, +40);
            texture = LoadTextureFromImage(image);
        }

        //        GuiSetTooltip("Set the random seed.");
        randomSeed = GuiSlider(Rectangle{ screenWidth - leftPadding,screenHeight - smallPadding - 7*padding, buttonWidth/2, 15},
                               "Seed", TextFormat("%i", (int)randomSeed),randomSeed, 1, 666);
//        GuiSetTooltip("Generate art.");

        comboBoxActive = GuiComboBox(Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 9*padding, buttonWidth+35, buttonHeight},
                                     "MOSAIC;CANDY;UDINY", comboBoxActive);
        animate=(GuiCheckBox(Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 8*padding, 20, 20}, "Animate", animate));
        static auto frameNumber=0;
        if (animate) {
            TraceLog(LOG_INFO, "Active: %i", comboBoxActive);
//            PlaySound(clickSound);
            TraceLog(LOG_INFO, "ncnnRay: Animate");
//            double r = ((double) rand() / (RAND_MAX));
//            moduleUdnie.train(); // On purpose to introduce randomness
            if (comboBoxActive +1==1) {
                
                texture = LoadTextureFromImage(image);
                DrawTextureEx(texture, Vector2{screenWidth / 2 - (float) texture.width * imageScale / 2,
                                               screenHeight / 2 - (float) texture.height * imageScale / 2},0.0f,imageScale, WHITE);
            }
            else if (comboBoxActive+1==2) {
                
                texture = LoadTextureFromImage(image);
                DrawTextureEx(texture, Vector2{screenWidth / 2 - (float) texture.width * imageScale / 2,
                                               screenHeight / 2 - (float) texture.height * imageScale / 2},0.0f,imageScale, WHITE);
            }

            else if (comboBoxActive+1==3) {
                
                texture = LoadTextureFromImage(image);
                DrawTextureEx(texture, Vector2{screenWidth / 2 - (float) texture.width * imageScale / 2,
                                               screenHeight / 2 - (float) texture.height * imageScale / 2},0.0f,imageScale, WHITE);
            }
            
            frameNumber++;
            statusText = TextFormat("Animating on GPU# %i, frame#: %i", frameNumber);
            GuiStatusBar(statusBarRect, statusText);
        }

//        GuiUnlock();
        if (GuiButton(Rectangle{screenWidth - leftPadding, screenHeight - smallPadding - 12 * padding, buttonWidth, buttonHeight},"SAVE")) {
            PlaySound(saveImageSound);
            TraceLog(LOG_INFO, "ncnnRay: save image");
            windowBoxActive = true;
        }

        GuiEnable();
        //-----------------------------------------------------------------------------
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
        //-----------------------------------------------------------------------------

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    UnloadImage(image);
    UnloadTexture(texture);

    CloseWindow();        // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

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

void handleDroppedFiles(const int screenWidth, const int screenHeight, int &pixelFormatActive, Image &image,
                        Texture2D &texture, bool &imageLoaded, float &imageScale) {
    if (IsFileDropped()) {
        int fileCount = 0;
        char **droppedFiles = GetDroppedFiles(&fileCount);

        if (fileCount == 1) {
            if (IsFileExtension(droppedFiles[0], ".png")) {
                TraceLog(LOG_INFO, "ncnnRay: image");
                Image imTemp = LoadImage(droppedFiles[0]);
                if (imTemp.data != nullptr) {
                    UnloadImage(image);
                    image = imTemp;
                    UnloadTexture(texture);
                    texture = LoadTextureFromImage(image);
                    imageLoaded = true;
                    pixelFormatActive = UNCOMPRESSED_R8G8B8A8;
                    if (texture.height > texture.width)
                        imageScale = (float) (screenHeight - 100) / (float) texture.height;
                    else imageScale = (float) (screenWidth - 100) / (float) texture.width;
                }
            }
        }
        ClearDroppedFiles();
    }
}

//int main(int argc, char *argv[]) {
//
//    bool use_vulkan_compute = true;
//    int gpu_device = 0;
//
//    g_blob_pool_allocator.clear();
//    g_workspace_pool_allocator.clear();
//
//#if NCNN_VULKAN
//    if (use_vulkan_compute) {
//        std::cout << "Using vulkan?: " << use_vulkan_compute << std::endl;
//
//        g_vkdev = ncnn::get_gpu_device(gpu_device);
//        g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
//        g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
//
//        g_blob_vkallocator->clear();
//        g_staging_vkallocator->clear();
//    }
//#endif // NCNN_VULKAN
//
//    ncnn::Option opt = optGPU(use_vulkan_compute, gpu_device);
//    std::string model_path = ".";
//    std::string fileName = "manga.png";
//
//    LFFD lffd1(model_path, 5, 0, opt, g_vkdev);
////    lffd1.detectFacesAndExportImage(fileName);
//
//    const int screenWidth = 1200;
//    const int screenHeight = 800;
//    Image image = LoadImage(fileName.c_str());   // Loaded in CPU memory (RAM)
//    lffd1.detectFacesAndDrawOnImage(image);
//    ImageResize(&image, image.width / 2, image.height / 2);
//
//    InitWindow(screenWidth, screenHeight, "random values (c++17)");
//    SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
//
//    while (!WindowShouldClose())    // Detect window close button or ESC key
//    {
////        ImageFormat(&image, UNCOMPRESSED_R8G8B8A8);
//        Texture2D texture = LoadTextureFromImage(image);
//        DrawTexture(texture, screenHeight / 2 - texture.width / 2, screenHeight / 2 - texture.height / 2, WHITE);
//        BeginDrawing();
//        ClearBackground(RAYWHITE);
////            DrawText("Generate a random value on the GPU/CPU using PyTorch", 30, 100, 20, MAROON);
//        EndDrawing();
//        UnloadTexture(texture);
//    }
//
//    CloseWindow();        // Close window and OpenGL context
//    return 0;
//}


