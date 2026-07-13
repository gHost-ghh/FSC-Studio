set(_onnxruntime_roots "")

if(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64" OR
   MSVC_CXX_ARCHITECTURE_ID STREQUAL "ARM64" OR
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
    set(_onnxruntime_rid "win-arm64")
else()
    set(_onnxruntime_rid "win-x64")
endif()

if(DEFINED ONNXRUNTIME_ROOT)
    list(APPEND _onnxruntime_roots "${ONNXRUNTIME_ROOT}")
endif()
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    list(APPEND _onnxruntime_roots "$ENV{ONNXRUNTIME_ROOT}")
endif()
list(APPEND _onnxruntime_roots "${CMAKE_CURRENT_LIST_DIR}/../../../.deps/onnxruntime")

find_path(OnnxRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${_onnxruntime_roots}
    PATH_SUFFIXES include build/native/include
)

find_library(OnnxRuntime_LIBRARY
    NAMES onnxruntime
    PATHS ${_onnxruntime_roots}
    PATH_SUFFIXES lib "runtimes/${_onnxruntime_rid}/native"
)

find_file(OnnxRuntime_RUNTIME_LIBRARY
    NAMES onnxruntime.dll
    PATHS ${_onnxruntime_roots}
    PATH_SUFFIXES lib "runtimes/${_onnxruntime_rid}/native"
)

find_path(OnnxRuntime_DML_INCLUDE_DIR
    NAMES dml_provider_factory.h
    PATHS ${_onnxruntime_roots}
    PATH_SUFFIXES include build/native/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntime
    REQUIRED_VARS OnnxRuntime_INCLUDE_DIR OnnxRuntime_LIBRARY OnnxRuntime_RUNTIME_LIBRARY
)

if(OnnxRuntime_FOUND AND NOT TARGET OnnxRuntime::OnnxRuntime)
    get_filename_component(_onnxruntime_runtime_dir "${OnnxRuntime_RUNTIME_LIBRARY}" DIRECTORY)
    file(GLOB OnnxRuntime_PROVIDER_RUNTIME_LIBRARIES
        "${_onnxruntime_runtime_dir}/onnxruntime_providers_*.dll"
    )
    list(FILTER OnnxRuntime_PROVIDER_RUNTIME_LIBRARIES EXCLUDE REGEX "onnxruntime_providers_tensorrt\\.dll$")
    file(GLOB OnnxRuntime_QNN_RUNTIME_LIBRARIES
        "${_onnxruntime_runtime_dir}/Qnn*.dll"
        "${_onnxruntime_runtime_dir}/libQnn*.so"
        "${_onnxruntime_runtime_dir}/libqnn*.cat"
    )
    set(OnnxRuntime_RUNTIME_LIBRARIES
        "${OnnxRuntime_RUNTIME_LIBRARY}"
        ${OnnxRuntime_PROVIDER_RUNTIME_LIBRARIES}
        ${OnnxRuntime_QNN_RUNTIME_LIBRARIES}
    )
    list(REMOVE_DUPLICATES OnnxRuntime_RUNTIME_LIBRARIES)

    foreach(provider_library IN LISTS OnnxRuntime_PROVIDER_RUNTIME_LIBRARIES)
        get_filename_component(provider_name "${provider_library}" NAME)
        if(provider_name STREQUAL "onnxruntime_providers_cuda.dll")
            set(OnnxRuntime_HAS_CUDA TRUE)
        elseif(provider_name STREQUAL "onnxruntime_providers_qnn.dll")
            set(OnnxRuntime_HAS_QNN TRUE)
        endif()
    endforeach()
    if(OnnxRuntime_QNN_RUNTIME_LIBRARIES)
        set(OnnxRuntime_HAS_QNN TRUE)
    endif()

    add_library(OnnxRuntime::OnnxRuntime UNKNOWN IMPORTED)
    set(_onnxruntime_include_dirs "${OnnxRuntime_INCLUDE_DIR}")
    if(OnnxRuntime_DML_INCLUDE_DIR)
        list(APPEND _onnxruntime_include_dirs "${OnnxRuntime_DML_INCLUDE_DIR}")
        set(OnnxRuntime_HAS_DML TRUE)
    endif()
    set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
        IMPORTED_LOCATION "${OnnxRuntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${_onnxruntime_include_dirs}"
    )

    message(STATUS "ONNX Runtime RID: ${_onnxruntime_rid}")
    message(STATUS "ONNX Runtime providers: DML=${OnnxRuntime_HAS_DML}; CUDA=${OnnxRuntime_HAS_CUDA}; QNN=${OnnxRuntime_HAS_QNN}")
endif()
