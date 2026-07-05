cmake_minimum_required(VERSION 3.28)

project(FscStudioNative VERSION 0.1.0 LANGUAGES CXX)

option(FSC_BUILD_QT_APP "Build the Qt desktop shell" OFF)
option(FSC_CORE_ONLY "Build dependency-light algorithm core only" OFF)
option(FSC_ENABLE_ONNX "Build ONNX Runtime based vision inference" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT FSC_CORE_ONLY)
    find_package(SQLite3 REQUIRED)
    find_package(nlohmann_json CONFIG REQUIRED)
    if(TARGET SQLite3::SQLite3)
        set(FSC_SQLITE_TARGET SQLite3::SQLite3)
    else()
        set(FSC_SQLITE_TARGET SQLite::SQLite3)
    endif()
endif()

add_library(FscCore STATIC
    src/core/IdentityGallery.cpp
    src/core/Search.cpp
    src/core/VectorMath.cpp
)
target_include_directories(FscCore PUBLIC include)
target_compile_definitions(FscCore PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

add_library(FscVision STATIC
    src/vision/FaceGeometry.cpp
    src/vision/Image.cpp
    src/vision/ModelPaths.cpp
)
target_include_directories(FscVision PUBLIC include)
target_compile_definitions(FscVision PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(FscVision PUBLIC FscCore)
if(FSC_ENABLE_ONNX)
    find_package(OnnxRuntime REQUIRED)
    target_sources(FscVision PRIVATE src/vision/InsightFaceEngine.cpp)
    target_sources(FscVision PRIVATE src/vision/OnnxRuntimeSession.cpp)
    target_compile_definitions(FscVision PUBLIC FSC_ENABLE_ONNX=1)
    target_link_libraries(FscVision PUBLIC OnnxRuntime::OnnxRuntime)
endif()

if(NOT FSC_CORE_ONLY)
    target_sources(FscCore PRIVATE src/core/Database.cpp)
    target_link_libraries(FscCore
        PUBLIC
            ${FSC_SQLITE_TARGET}
            nlohmann_json::nlohmann_json
    )
    add_executable(fsc_native_probe apps/fsc_native_probe/main.cpp)
    target_link_libraries(fsc_native_probe PRIVATE FscCore FscVision)
endif()

add_executable(fsc_vision_probe apps/fsc_vision_probe/main.cpp)
target_link_libraries(fsc_vision_probe PRIVATE FscVision)
if(FSC_ENABLE_ONNX)
    add_custom_command(TARGET fsc_vision_probe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OnnxRuntime_RUNTIME_LIBRARY}"
            "$<TARGET_FILE_DIR:fsc_vision_probe>"
        COMMENT "Copying ONNX Runtime DLL next to fsc_vision_probe"
    )
endif()

enable_testing()
add_executable(fsc_core_tests tests/core_tests.cpp)
target_link_libraries(fsc_core_tests PRIVATE FscCore FscVision)
add_test(NAME fsc_core_tests COMMAND fsc_core_tests)

if(FSC_BUILD_QT_APP)
    find_package(Qt6 REQUIRED COMPONENTS Widgets)
    add_executable(FscStudioQt WIN32 apps/FscAppQt/main.cpp)
    target_link_libraries(FscStudioQt PRIVATE FscCore FscVision Qt6::Widgets)
endif()
