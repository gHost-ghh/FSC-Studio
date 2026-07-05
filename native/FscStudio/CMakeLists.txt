cmake_minimum_required(VERSION 3.28)

project(FscStudioNative VERSION 0.1.0 LANGUAGES CXX)

option(FSC_BUILD_QT_APP "Build the Qt desktop shell" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SQLite3 REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_library(FscCore STATIC
    src/core/Database.cpp
    src/core/IdentityGallery.cpp
    src/core/Search.cpp
    src/core/VectorMath.cpp
)
target_include_directories(FscCore PUBLIC include)
target_link_libraries(FscCore
    PUBLIC
        SQLite::SQLite3
        nlohmann_json::nlohmann_json
)
target_compile_definitions(FscCore PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

add_executable(fsc_native_probe apps/fsc_native_probe/main.cpp)
target_link_libraries(fsc_native_probe PRIVATE FscCore)

enable_testing()
add_executable(fsc_core_tests tests/core_tests.cpp)
target_link_libraries(fsc_core_tests PRIVATE FscCore)
add_test(NAME fsc_core_tests COMMAND fsc_core_tests)

if(FSC_BUILD_QT_APP)
    find_package(Qt6 REQUIRED COMPONENTS Widgets)
    add_executable(FscStudioQt WIN32 apps/FscAppQt/main.cpp)
    target_link_libraries(FscStudioQt PRIVATE FscCore Qt6::Widgets)
endif()
