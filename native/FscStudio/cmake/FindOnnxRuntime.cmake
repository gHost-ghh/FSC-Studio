set(_onnxruntime_roots "")

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
    PATH_SUFFIXES lib runtimes/win-x64/native
)

find_file(OnnxRuntime_RUNTIME_LIBRARY
    NAMES onnxruntime.dll
    PATHS ${_onnxruntime_roots}
    PATH_SUFFIXES lib runtimes/win-x64/native
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
endif()
