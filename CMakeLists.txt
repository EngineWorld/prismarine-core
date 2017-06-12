cmake_minimum_required (VERSION 3.0)
set (PROJECT_NAME paper-render)
set (APPLICATION_NAME paper-render)

project (${PROJECT_NAME})

set(SOURCES_LIST 
    "./example/*.cpp"

    #include
    "./include/*.hpp"
    "./include/*.hh"
    "./include/*.h"
    
    #include
    "./external/include/*.hpp"
    "./external/include/*.hh"
    "./external/include/*.h"
    
    #GLAD
    "./include/glad/*.h"
    "./include/glad/*.cpp"
    
    #ray-tracer
    "./include/tracer/*.hpp"
    "./include/tracer/*.inl"
    
    #shaders
    "./shaders-sdk/include/*.*"
    "./shaders-sdk/render/*.*"
    "./shaders-sdk/hlbvh/*.*"
    "./shaders-sdk/tools/*.*"
)

set (DEFINES
    -D_USE_MATH_DEFINES 
    -DNOMINMAX 
    -DGLM_FORCE_SWIZZLE 
    -DGLFW_INCLUDE_NONE 
    -DUSE_FREEIMAGE 
    -DASSIMP_SUPPORT 
    -DGLAD_SUPPORT
)

set (DEP_DIR "${PROJECT_SOURCE_DIR}")
set (LIB_DIR "${DEP_DIR}/external/lib"     "${DEP_DIR}/lib"     "${CMAKE_MODULE_PATH}/lib")
set (INC_DIR "${DEP_DIR}/external/include" "${DEP_DIR}/include" "${CMAKE_MODULE_PATH}/include" "${PROJECT_SOURCE_DIR}")

set (CMAKE_MODULE_PATH 
    "${DEP_DIR}/modules" 
    "${CMAKE_MODULE_PATH}/modules" 
    "${PROJECT_SOURCE_DIR}/modules/" )

find_package(OpenGL REQUIRED)

set (LIBS 
    ${OPENGL_LIBRARIES}
    glfw3 
    FreeImage 
    assimp
)

set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_EXTENSIONS ON)
set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Ofast)
add_compile_options(-std=c++17 /Ox /Ot /Oi /Oy /GT /GL /arch:AVX2)

set (INCLUDE_LIST 
    ${INC_DIR}
    ${OPENGL_INCLUDE_DIR}
)

add_definitions(${DEFINES})
link_directories(${LIB_DIR})
include_directories(${INCLUDE_LIST})
file (GLOB RSOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_LIST})
add_executable(${APPLICATION_NAME} example/main.cpp ${RSOURCES})
target_link_libraries(${APPLICATION_NAME} ${LIBS})

foreach(source IN LISTS RSOURCES)
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach()
