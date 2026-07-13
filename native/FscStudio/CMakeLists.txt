cmake_minimum_required(VERSION 3.28)

project(FscStudioNative VERSION 1.0.0 LANGUAGES CXX)

option(FSC_BUILD_QT_APP "Build the Qt desktop shell" OFF)
option(FSC_CORE_ONLY "Build dependency-light algorithm core only" OFF)
option(FSC_ENABLE_ONNX "Build ONNX Runtime based vision inference" OFF)
option(FSC_ENABLE_OPENCV "Build OpenCV based native camera capture" OFF)
set(FSC_FACE_MESH_MODEL_PATH "" CACHE FILEPATH "Path to the ONNX MediaPipe 478-point face-landmark model")
set(FSC_SQLITE_RUNTIME_PATH "" CACHE FILEPATH "Path to the target-architecture sqlite3.dll")
set(FSC_ZLIB_RUNTIME_PATH "" CACHE FILEPATH "Path to the target-architecture zlib runtime DLL used by OpenCV")

if(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64" OR
   MSVC_CXX_ARCHITECTURE_ID STREQUAL "ARM64" OR
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
    set(FSC_TARGET_ARM64 TRUE)
else()
    set(FSC_TARGET_ARM64 FALSE)
endif()

if(NOT FSC_FACE_MESH_MODEL_PATH)
    foreach(candidate
        "${CMAKE_SOURCE_DIR}/../../model/mediapipe/face_landmarks_detector.onnx"
        "${CMAKE_SOURCE_DIR}/../../.deps/mediapipe-models/face_landmarks_detector_tf2onnx.onnx")
        if(EXISTS "${candidate}")
            set(FSC_FACE_MESH_MODEL_PATH "${candidate}")
            break()
        endif()
    endforeach()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT FSC_CORE_ONLY OR FSC_BUILD_QT_APP)
    find_package(SQLite3 REQUIRED)
    find_package(nlohmann_json CONFIG REQUIRED)
    if(TARGET SQLite3::SQLite3)
        set(FSC_SQLITE_TARGET SQLite3::SQLite3)
    else()
        set(FSC_SQLITE_TARGET SQLite::SQLite3)
    endif()
    if(NOT FSC_SQLITE_RUNTIME_PATH AND SQLite3_LIBRARY AND FSC_TARGET_ARM64)
        get_filename_component(_fsc_sqlite_lib_dir "${SQLite3_LIBRARY}" DIRECTORY)
        get_filename_component(_fsc_sqlite_prefix "${_fsc_sqlite_lib_dir}" DIRECTORY)
        set(_fsc_sqlite_candidates
            "${_fsc_sqlite_prefix}/bin/sqlite3.dll"
            "${CMAKE_SOURCE_DIR}/../../.deps/sqlite-arm64/bin/sqlite3.dll")
        foreach(candidate IN LISTS _fsc_sqlite_candidates)
            if(EXISTS "${candidate}")
                set(FSC_SQLITE_RUNTIME_PATH "${candidate}")
                break()
            endif()
        endforeach()
        unset(_fsc_sqlite_candidates)
        unset(_fsc_sqlite_lib_dir)
        unset(_fsc_sqlite_prefix)
    endif()
    if(NOT FSC_ZLIB_RUNTIME_PATH AND FSC_TARGET_ARM64)
        foreach(candidate
            "${CMAKE_SOURCE_DIR}/../../.deps/zlib-arm64/bin/z.dll")
            if(EXISTS "${candidate}")
                set(FSC_ZLIB_RUNTIME_PATH "${candidate}")
                break()
            endif()
        endforeach()
    endif()
endif()

add_library(FscCore STATIC
    src/core/FileHash.cpp
    src/core/IdentityGallery.cpp
    src/core/PathEncoding.cpp
    src/core/Search.cpp
    src/core/VectorMath.cpp
)
target_include_directories(FscCore PUBLIC include)
target_compile_definitions(FscCore PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
if(WIN32)
    target_link_libraries(FscCore PUBLIC Bcrypt)
endif()

add_library(FscVision STATIC
    src/vision/FaceGeometry.cpp
    src/vision/Image.cpp
    src/vision/ModelPaths.cpp
)
target_include_directories(FscVision PUBLIC include)
target_compile_definitions(FscVision PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(FscVision PUBLIC FscCore)
if(WIN32)
    target_link_libraries(FscVision PUBLIC Windowscodecs Ole32)
endif()
if(FSC_ENABLE_ONNX)
    find_package(OnnxRuntime REQUIRED)
    target_sources(FscVision PRIVATE src/vision/InsightFaceEngine.cpp)
    target_sources(FscVision PRIVATE src/vision/OnnxRuntimeSession.cpp)
    target_sources(FscVision PRIVATE src/vision/RuntimeProvider.cpp)
    target_compile_definitions(FscVision PUBLIC FSC_ENABLE_ONNX=1)
    if(OnnxRuntime_HAS_DML)
        target_compile_definitions(FscVision PUBLIC FSC_ONNXRUNTIME_HAS_DML=1)
    endif()
    if(OnnxRuntime_HAS_CUDA)
        target_compile_definitions(FscVision PUBLIC FSC_ONNXRUNTIME_HAS_CUDA=1)
    endif()
    if(OnnxRuntime_HAS_QNN)
        target_compile_definitions(FscVision PUBLIC FSC_ONNXRUNTIME_HAS_QNN=1)
    endif()
    target_link_libraries(FscVision PUBLIC OnnxRuntime::OnnxRuntime)
    function(fsc_copy_onnx_runtime target_name)
        foreach(runtime_library IN LISTS OnnxRuntime_RUNTIME_LIBRARIES)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${runtime_library}"
                    "$<TARGET_FILE_DIR:${target_name}>"
                COMMENT "Copying ${runtime_library} next to ${target_name}"
            )
        endforeach()
    endfunction()
endif()

add_library(FscMesh STATIC
    src/mesh/FaceMesh.cpp
)
target_include_directories(FscMesh PUBLIC include)
target_compile_definitions(FscMesh PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(FscMesh PUBLIC FscVision)

function(fsc_copy_face_mesh_model target_name)
    if(FSC_FACE_MESH_MODEL_PATH AND EXISTS "${FSC_FACE_MESH_MODEL_PATH}")
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target_name}>/models/mediapipe"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FSC_FACE_MESH_MODEL_PATH}"
                "$<TARGET_FILE_DIR:${target_name}>/models/mediapipe/face_landmarks_detector.onnx"
            COMMENT "Copying ONNX dense face mesh model next to ${target_name}"
        )
    else()
        message(WARNING "FSC_FACE_MESH_MODEL_PATH is not set; Dense Mesh generation requires face_landmarks_detector.onnx under models/mediapipe.")
    endif()
endfunction()

if(NOT FSC_CORE_ONLY OR FSC_BUILD_QT_APP)
    target_sources(FscCore PRIVATE src/core/Database.cpp)
    target_link_libraries(FscCore
        PUBLIC
            ${FSC_SQLITE_TARGET}
            nlohmann_json::nlohmann_json
    )
    if(FSC_ENABLE_ONNX)
        add_library(FscLegacy STATIC
            src/legacy/LegacyDtb.cpp
        )
        target_include_directories(FscLegacy PUBLIC include)
        target_compile_definitions(FscLegacy PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
        target_link_libraries(FscLegacy PUBLIC FscCore FscVision)
    endif()
    if(NOT FSC_CORE_ONLY)
        add_executable(fsc_native_probe apps/fsc_native_probe/main.cpp)
        target_link_libraries(fsc_native_probe PRIVATE FscCore FscVision FscMesh)
        if(TARGET FscLegacy)
            target_link_libraries(fsc_native_probe PRIVATE FscLegacy)
        endif()
        fsc_copy_face_mesh_model(fsc_native_probe)
        if(FSC_ENABLE_ONNX)
            fsc_copy_onnx_runtime(fsc_native_probe)
        endif()
    endif()
endif()

add_executable(fsc_vision_probe apps/fsc_vision_probe/main.cpp)
target_link_libraries(fsc_vision_probe PRIVATE FscVision)
if(FSC_ENABLE_ONNX)
    fsc_copy_onnx_runtime(fsc_vision_probe)
endif()

enable_testing()
add_executable(fsc_core_tests tests/core_tests.cpp)
target_link_libraries(fsc_core_tests PRIVATE FscCore FscVision FscMesh)
if(FSC_CORE_ONLY)
    target_compile_definitions(fsc_core_tests PRIVATE FSC_CORE_ONLY=1)
endif()
if(TARGET FscLegacy)
    target_link_libraries(fsc_core_tests PRIVATE FscLegacy)
endif()
add_test(NAME fsc_core_tests COMMAND fsc_core_tests)

if(FSC_BUILD_QT_APP)
    find_package(Qt6 REQUIRED COMPONENTS Widgets Concurrent)
    function(fsc_copy_qt_runtime target_name)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target_name}>
                $<TARGET_FILE_DIR:${target_name}>
            COMMAND_EXPAND_LISTS
            COMMENT "Copying linked runtime DLLs next to ${target_name}"
        )
        foreach(plugin_target IN ITEMS Qt6::QWindowsIntegrationPlugin Qt6::QMinimalIntegrationPlugin)
            if(TARGET ${plugin_target})
                add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target_name}>/platforms"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "$<TARGET_FILE:${plugin_target}>"
                        "$<TARGET_FILE_DIR:${target_name}>/platforms"
                    COMMENT "Copying Qt platform plugin ${plugin_target} next to ${target_name}"
                )
            endif()
        endforeach()
        foreach(plugin_target IN ITEMS Qt6::QJpegPlugin Qt6::QPngPlugin Qt6::QGifPlugin Qt6::QICOPlugin)
            if(TARGET ${plugin_target})
                add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target_name}>/imageformats"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "$<TARGET_FILE:${plugin_target}>"
                        "$<TARGET_FILE_DIR:${target_name}>/imageformats"
                    COMMENT "Copying Qt image format plugin ${plugin_target} next to ${target_name}"
                )
            endif()
        endforeach()
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/apps/FscAppQt/qt.conf"
                "$<TARGET_FILE_DIR:${target_name}>/qt.conf"
            COMMENT "Copying Qt runtime configuration next to ${target_name}"
        )
    endfunction()

    add_executable(FscStudioQt WIN32 apps/FscAppQt/main.cpp)
    target_link_libraries(FscStudioQt PRIVATE FscCore FscVision FscMesh Qt6::Widgets Qt6::Concurrent)
    if(WIN32)
        target_link_libraries(FscStudioQt PRIVATE Shell32)
    endif()
    if(TARGET FscLegacy)
        target_link_libraries(FscStudioQt PRIVATE FscLegacy)
    endif()
    if(MSVC)
        target_compile_options(FscStudioQt PRIVATE /utf-8)
    endif()
    fsc_copy_qt_runtime(FscStudioQt)
    fsc_copy_face_mesh_model(FscStudioQt)
    if(FSC_ENABLE_ONNX)
        fsc_copy_onnx_runtime(FscStudioQt)
    endif()
    foreach(runtime_file IN ITEMS "${FSC_SQLITE_RUNTIME_PATH}" "${FSC_ZLIB_RUNTIME_PATH}")
        if(runtime_file AND EXISTS "${runtime_file}")
            add_custom_command(TARGET FscStudioQt POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${runtime_file}"
                    "$<TARGET_FILE_DIR:FscStudioQt>"
                COMMENT "Copying ${runtime_file} next to FscStudioQt"
            )
        endif()
    endforeach()
    if(FSC_ENABLE_OPENCV)
        find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)
        target_compile_definitions(FscStudioQt PRIVATE FSC_ENABLE_OPENCV=1)
        target_link_libraries(FscStudioQt PRIVATE ${OpenCV_LIBS})
        foreach(opencv_target IN ITEMS opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs)
            if(TARGET ${opencv_target})
                add_custom_command(TARGET FscStudioQt POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "$<TARGET_FILE:${opencv_target}>"
                        "$<TARGET_FILE_DIR:FscStudioQt>"
                    COMMENT "Copying ${opencv_target} next to FscStudioQt"
                )
            endif()
        endforeach()
    endif()
endif()
