cmake_minimum_required(VERSION 3.11) # FetchContent is available in 3.11+
#project(music_tiles)

project(ncnnRay
        DESCRIPTION
        "ncnnRay"
        LANGUAGES
        CXX C
        )

include(FetchContent)

if (EMSCRIPTEN )
    set(PLATFORM_WEB ON)
endif()

option(GPU_SUPPORT "Enable GPU support (Vulkan)" ON)
message("-- NCNN_VULKAN flag is ${NCNN_VULKAN}")
if (${NCNN_VULKAN})
    message("-- AUTO-ENABLING GPU_SUPPORT")
    set(GPU_SUPPORT ON)
endif()

if (GPU_SUPPORT)
    message("-- GPU support is ENABLED")
    find_package(Vulkan) # REQUIRES ncnn to be built with vulkan
    if (${VULKAN_FOUND})
        message("-- Vulkan found.")
    else()
        message("-- ERROR: AUTO-DISABLING GPU_SUPPORT, because Vulkan was not found")
        set(GPU_SUPPORT OFF)
    endif()
else()
    message("-- GPU support is DISABLED")
endif()


# shlomo
if (NOT EMSCRIPTEN)
    set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin64)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
    set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
else()
    set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/web)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
    set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
endif()

if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo
MinSizeRel."
            FORCE )
endif()

set(CMAKE_BUILD_TYPE Release)
set(CXX_RELEASE_FLAGS = -O3 -march=native )


if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "*** Have SIZEOF void * = 8, so 64-bit")
    set( IS_64_BIT 1 )
else ()
	if (NOT EMSCRIPTEN)
		message(STATUS "*** SIZEOF void * != 8, so not 64-bit")
		message(FATAL_ERROR "Only 64-bit i supported")
	endif ()	
endif ()


if (MSVC)
    # Disable annoying MSVC warnings (all targets)
    add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
    # Parallel build on MSVC (all targets)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Set this to the minimal version you want to support
find_package(raylib 3.0 QUIET) # Let CMake search for a raylib-config.cmake

if (NOT raylib_FOUND) # If there's none, fetch and build raylib
    include(FetchContent)
    FetchContent_Declare(
            raylib
            URL https://github.com/raysan5/raylib/archive/3.0.0.tar.gz
    )
    FetchContent_GetProperties(raylib)
    if (NOT raylib_POPULATED) # Have we downloaded raylib yet?
        set(FETCHCONTENT_QUIET NO)
        FetchContent_Populate(raylib)
        set(BUILD_GAMES OFF CACHE BOOL "" FORCE) # don't build the supplied examples
        set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # don't build the supplied examples
        # build raylib
        add_subdirectory(${raylib_SOURCE_DIR} ${raylib_BINARY_DIR})
    endif()
endif()

#set (NCNN_BUILD_EXAMPLES ON)
#set (NCNN_SYSTEM_GLSLANG ON)

add_subdirectory(3rdparty/ncnn ncnn_build)
#include_directories(3rdparty/ncnn/src)


################## EXAMPLE 006 ########################
## TARGET
set(EXAMPLE_001_WASM ${PROJECT_NAME}_wasm_test)
#add_executable(${EXAMPLE_005_EXE} src/${EXAMPLE_005_EXE}.cpp)

if (EMSCRIPTEN)
#    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
#    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
#    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
#    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
#    set(EMSCRIPTEN_LINK_FLAGS "${EMSCRIPTEN_LINK_FLAGS} --memory-init-file 0 -s TOTAL_MEMORY=512MB -s WASM_MEM_MAX=2GB -s ALLOW_MEMORY_GROWTH=1 -s MALLOC=emmalloc")
#    set(FLAGS "-o ${EXAMPLE_001_WASM}.html -s -O3 -s ALLOW_MEMORY_GROWTH=1 --embed-file ./resources/cat.png@cat.png")
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
#    set(CMAKE_EXECUTABLE_SUFFIX ".wasm")
#    set (CMAKE_CXX_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s FETCH=1 -s ASSERTIONS=1 -s DISABLE_EXCEPTION_CATCHING=0 -s ERROR_ON_UNDEFINED_SYMBOLS=0 --embed-file ./resources/cat.png@cat.png")
#    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/resources/index-tmpl.html  ${CMAKE_SOURCE_DIR}/web/index.html @ONLY)
#    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/resources/style.css ${CMAKE_SOURCE_DIR}/web/style.css @ONLY)
    add_executable(${EXAMPLE_001_WASM} src/${EXAMPLE_001_WASM}.cpp
            include/ncnnRay.hpp
            include/raygui/raygui.h
            include/raygui/ricons.h
            include/ncnnRay.hpp
            src/models/neural.cpp
            src/models/neural.h
            src/models/LFFD.cpp
            src/models/LFFD.h
            src/models/FaceDetector.h
            src/models/FaceDetector.cpp
            )
    target_link_libraries(${EXAMPLE_001_WASM} raylib  ncnn)
    target_include_directories(
            ${EXAMPLE_001_WASM} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/src"
            PUBLIC "c:/emsdk/upstream/emscripten/system/include/emscripten/"
            "c:/emsdk/upstream/emscripten/system/include/"
    )

    #    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index-tmpl.html ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/index.html @ONLY)
    #    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/style.css ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/style.css @ONLY)

else()
    add_executable(${EXAMPLE_001_WASM} src/${EXAMPLE_001_WASM}.cpp
            include/ncnnRay.hpp
            include/raygui/raygui.h
            include/raygui/ricons.h
            include/ncnnRay.hpp
            src/models/neural.cpp
            src/models/neural.h
            src/models/LFFD.cpp
            src/models/LFFD.h
            src/models/FaceDetector.h
            src/models/FaceDetector.cpp
            )
    set(raylib_VERBOSE 1)
    target_link_libraries(${EXAMPLE_001_WASM} raylib  ncnn)
    target_include_directories(
            ${EXAMPLE_001_WASM} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/src"
            PUBLIC "c:/emsdk/upstream/emscripten/system/include/emscripten/"
            "c:/emsdk/upstream/emscripten/system/include/"
    )
    set_target_properties(${EXAMPLE_001_WASM} PROPERTIES
            CXX_STANDARD 14
#            -fPIC
            CXX_STANDARD_REQUIRED YES)
endif ()

################## EXAMPLE 006 ########################


################### EXAMPLE 006 ########################
### TARGET
#set(EXAMPLE_001_EXE ${PROJECT_NAME}_read_image)
##add_executable(${EXAMPLE_005_EXE} src/${EXAMPLE_005_EXE}.cpp)
#
#if (EMSCRIPTEN)
#    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
#    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
#    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
#    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
#
#    set(FLAGS "-o ${EXAMPLE_001_EXE}.html -s -O3")
#    set(CMAKE_EXECUTABLE_SUFFIX ".html")
#    set (CMAKE_CXX_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s FETCH=1 -s ASSERTIONS=1 -s DISABLE_EXCEPTION_CATCHING=0")
#
#    add_executable(${EXAMPLE_001_EXE} src/${EXAMPLE_001_EXE}.cpp
#            include/ncnnRay.hpp
#            include/raygui/raygui.h
#            include/raygui/ricons.h
#            )
#    target_link_libraries(${EXAMPLE_001_EXE} raylib  ncnn)
#    target_include_directories(
#            ${EXAMPLE_001_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/src"
#            PUBLIC "c:/emsdk/upstream/emscripten/system/include/emscripten/"
#            "c:/emsdk/upstream/emscripten/system/include/"
#            )
#
#    #    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index-tmpl.html ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/index.html @ONLY)
#    #    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/style.css ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/style.css @ONLY)
#
#else()
#    add_executable(${EXAMPLE_001_EXE} src/${EXAMPLE_001_EXE}.cpp
#            include/ncnnRay.hpp
#            include/raygui/raygui.h
#            include/raygui/ricons.h
#            )
#    set(raylib_VERBOSE 1)
#    target_link_libraries(${EXAMPLE_001_EXE} raylib  ncnn)
#    target_include_directories(
#            ${EXAMPLE_001_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/src"
#            PUBLIC "c:/emsdk/upstream/emscripten/system/include/emscripten/"
#            "c:/emsdk/upstream/emscripten/system/include/"
#    )
#    set_target_properties(${EXAMPLE_001_EXE} PROPERTIES
#            CXX_STANDARD 14
#            CXX_STANDARD_REQUIRED YES)
#endif()

################# EXAMPLE 006 ########################


if (NOT EMSCRIPTEN)
    ################# EXAMPLE 006 ########################
    # TARGET
    set(EXAMPLE_002_EXE ${PROJECT_NAME}_lffd_face)
    #add_executable(${EXAMPLE_005_EXE} src/${EXAMPLE_005_EXE}.cpp)
    add_executable(${EXAMPLE_002_EXE} src/${EXAMPLE_002_EXE}.cpp
            src/models/LFFD.cpp src/models/LFFD.h)
    set(raylib_VERBOSE 1)
    target_link_libraries(${EXAMPLE_002_EXE} raylib  ncnn)
    #target_link_libraries(${PROJECT_NAME} m)
    target_include_directories(${EXAMPLE_002_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/src")
    set_target_properties(${EXAMPLE_002_EXE} PROPERTIES
            CXX_STANDARD 14
            CXX_STANDARD_REQUIRED YES)
    ################# EXAMPLE 006 ########################

    ################# EXAMPLE 006 ########################
    # TARGET
    set(EXAMPLE_003_EXE ${PROJECT_NAME}_studio_UI)
    #add_executable(${EXAMPLE_005_EXE} src/${EXAMPLE_005_EXE}.cpp)
    add_executable(${EXAMPLE_003_EXE} src/${EXAMPLE_003_EXE}.cpp
            src/models/LFFD.cpp
            src/models/LFFD.h
            src/models/FaceDetector.h
            src/models/FaceDetector.cpp
            src/models/neural.h
            src/models/neural.cpp

            )
    set(raylib_VERBOSE 1)
    target_link_libraries(${EXAMPLE_003_EXE} raylib  ncnn)
    #target_link_libraries(${PROJECT_NAME} m)
    target_include_directories(${EXAMPLE_003_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/src")
    set_target_properties(${EXAMPLE_003_EXE} PROPERTIES
            CXX_STANDARD 14
            CXX_STANDARD_REQUIRED YES)
    ################# EXAMPLE 006 ########################


    ################# EXAMPLE 006 ########################
    # TARGET
    set(EXAMPLE_009_EXE ${PROJECT_NAME}_video_UI)
    #set(EXAMPLE_009_EXE ${PROJECT_NAME}_becnh_gpu)
    add_executable(${EXAMPLE_009_EXE} src/${EXAMPLE_009_EXE}.cpp
    #add_executable(${EXAMPLE_009_EXE} src/models/ncnnRay_bench_gpu.cpp
            include/ncnnRay.hpp
            include/raygui/raygui.h
            include/raygui/ricons.h
            include/pl_mpeg/pl_mpeg.h
            src/models/LFFD.cpp
            src/models/LFFD.h
            src/models/neural.cpp
            src/models/neural.h
            src/models/FaceDetector.h
            src/models/FaceDetector.cpp
            )
    set(raylib_VERBOSE 1)
    target_link_libraries(${EXAMPLE_009_EXE} raylib  ncnn)
    #target_link_libraries(${PROJECT_NAME} m)
    target_include_directories(${EXAMPLE_009_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/src")
    set_target_properties(${EXAMPLE_009_EXE} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED YES)
    #target_compile_options(${EXAMPLE_009_EXE} PRIVATE /wd4267 /wd4244 /wd4996)
    ################# EXAMPLE 006 ########################


    ################# EXAMPLE 006 ########################
    # TARGET
    set(EXAMPLE_012_EXE ${PROJECT_NAME}_retinaface)
    #add_executable(${EXAMPLE_005_EXE} src/${EXAMPLE_005_EXE}.cpp)
    add_executable(${EXAMPLE_012_EXE} src/${EXAMPLE_012_EXE}.cpp
            src/models/FaceDetector.h src/models/FaceDetector.cpp
            )
    set(raylib_VERBOSE 1)
    target_link_libraries(${EXAMPLE_012_EXE} raylib  ncnn)
    #target_link_libraries(${PROJECT_NAME} m)
    target_include_directories(${EXAMPLE_012_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/src")
    set_target_properties(${EXAMPLE_012_EXE} PROPERTIES
            CXX_STANDARD 14
            CXX_STANDARD_REQUIRED YES)
    ################# EXAMPLE 006 ########################


    ################# EXAMPLE 006 ########################
    # TARGET
    set(EXAMPLE_0011_EXE ${PROJECT_NAME}_neural)
    #add_executable(${EXAMPLE_005_EXE} src/${EXAMPLE_005_EXE}.cpp)
    add_executable(${EXAMPLE_0011_EXE} src/${EXAMPLE_0011_EXE}.cpp
            include/ncnnRay.hpp
            src/models/neural.cpp
            src/models/neural.h
            )
    set(raylib_VERBOSE 1)
    target_link_libraries(${EXAMPLE_0011_EXE} raylib  ncnn)
    #target_link_libraries(${PROJECT_NAME} m)
    target_include_directories(${EXAMPLE_0011_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/src")
    set_target_properties(${EXAMPLE_0011_EXE} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED YES)
    #target_compile_options(${EXAMPLE_009_EXE} PRIVATE /wd4267 /wd4244 /wd4996)
    ################# EXAMPLE 006 ########################
endif()


#add_executable(ncnn_ultraface src/ultra.cpp src/UltraFace.cpp)
#target_link_libraries(ncnn_ultraface ncnn)

SET(ACTUAL_DIR  ${CMAKE_SOURCE_DIR}/bin64)
if (EMSCRIPTEN)
    SET(ACTUAL_DIR  ${CMAKE_SOURCE_DIR}/web)
endif()
#add_executable(ncnn_centerface src/ncnn_center.cpp src/ncnn_centerface.cpp src/ncnn_centerface.h)
#target_link_libraries(ncnn_centerface ncnn)
#file(COPY "resources/parrots.png" DESTINATION ${ACTUAL_DIR})
#file(COPY "resources/cat.png" DESTINATION ${ACTUAL_DIR})
#file(COPY "resources/big-cat.png" DESTINATION ${ACTUAL_DIR})
#file(COPY "resources/baboon.png" DESTINATION ${ACTUAL_DIR})
file(COPY "resources/faces.png" DESTINATION ${ACTUAL_DIR})
file(COPY "resources/faces01.png" DESTINATION ${ACTUAL_DIR})
#file(COPY "resources/windmill.png" DESTINATION ${ACTUAL_DIR})
file(COPY "resources/manga.png" DESTINATION ${ACTUAL_DIR})

# # Copy trained PTH files
#file(COPY "resources/centerface.bin" DESTINATION ${ACTUAL_DIR})
#file(COPY "resources/centerface.param" DESTINATION ${ACTUAL_DIR})

file(COPY "resources/symbol_10_320_20L_5scales_v2_deploy.param" DESTINATION ${ACTUAL_DIR})
file(COPY "resources/symbol_10_560_25L_8scales_v1_deploy.param" DESTINATION ${ACTUAL_DIR})

file(COPY "resources/train_10_320_20L_5scales_v2_iter_1000000.bin" DESTINATION ${ACTUAL_DIR})
file(COPY "resources/train_10_560_25L_8scales_v1_iter_1400000.bin" DESTINATION ${ACTUAL_DIR})

file(COPY "resources/retinaface.param" DESTINATION ${ACTUAL_DIR})
file(COPY "resources/retinaface.bin" DESTINATION ${ACTUAL_DIR})
file(COPY "resources/GameCube.ttf" DESTINATION ${ACTUAL_DIR})
#file(COPY "resources/EBGaramond08-Regular.otf" DESTINATION${ACTUAL_DIR})

#file(COPY "resources/save.ogg" DESTINATION ${ACTUAL_DIR})
#file(COPY "resources/click.ogg" DESTINATION ${ACTUAL_DIR})
#file(COPY "resources/bjork-all-is-full-of-love.mpg" DESTINATION${ACTUAL_DIR})

#file(COPY "resources/torch.rgs" DESTINATION${ACTUAL_DIR})
file(COPY "resources/torch2.rgs" DESTINATION ${ACTUAL_DIR})
file(COPY "resources/mosaic.bin" DESTINATION ${ACTUAL_DIR})
file(COPY "resources/candy.bin" DESTINATION ${ACTUAL_DIR})

#file(COPY "resources/ncnn.html" DESTINATION ${ACTUAL_DIR})
#file(COPY "resources/shell-ray.html" DESTINATION ${ACTUAL_DIR})
#file(COPY "resources/shell.html" DESTINATION ${ACTUAL_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/index.html DESTINATION ${ACTUAL_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/style.css DESTINATION ${ACTUAL_DIR})

file(COPY ${ACTUAL_DIR}/ncnnRay_wasm_test.js DESTINATION ${CMAKE_SOURCE_DIR}/)
file(COPY ${ACTUAL_DIR}/ncnnRay_wasm_test.wasm DESTINATION ${CMAKE_SOURCE_DIR}/)
file(COPY ${ACTUAL_DIR}/ncnnRay_wasm_test.data DESTINATION ${CMAKE_SOURCE_DIR}/)




