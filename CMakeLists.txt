cmake_minimum_required (VERSION 3.0)
set (PROJECT_NAME paper-render)
set (APPLICATION_NAME paper-render)

project (${PROJECT_NAME})

if (COMPILER_ID MATCHES "MSVC")
  set(MSVC_${lang}_ARCHITECTURE_ID "${ARCHITECTURE_ID}")
endif()

set(SOURCES_LIST 
    #include
    "./include/*.hpp"
    "./include/*.hh"
    "./include/*.h"
    
    #include
    "./external/include/*/*.hpp"
    "./external/include/*/*.hh"
    "./external/include/*/*.h"
    "./external/include/*.hpp"
    "./external/include/*.hh"
    "./external/include/*.h"
    
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
    -DNOMINMAX 
    -DGLM_FORCE_SWIZZLE 
    -DGLM_ENABLE_EXPERIMENTAL 
    -DGLFW_INCLUDE_NONE 
    -DUSE_FREEIMAGE 
    -DEXPERIMENTAL_GLTF
    -DTINYGLTF_IMPLEMENTATION
    -DSTB_IMAGE_IMPLEMENTATION
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
    assimp-vc140-mt
	glew32
)

set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_EXTENSIONS ON)

if(MSVC)
set (CMAKE_CXX_FLAGS "/EHsc /std:c++latest -DMSVC")
set (CMAKE_C_FLAGS "/EHsc")
else()
set (CMAKE_CXX_FLAGS "-std=c++1z")
set (CMAKE_C_FLAGS "")
endif()

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




add_executable(${APPLICATION_NAME}-physics example/physics.cpp ${RSOURCES})
target_link_libraries(${APPLICATION_NAME}-physics ${LIBS})

foreach(source IN LISTS RSOURCES)
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach()

