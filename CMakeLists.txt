cmake_minimum_required (VERSION 3.0)
set (PROJECT_NAME prismarine-examples)
set (APPLICATION_NAME prismarine-gltf)

project (${PROJECT_NAME})

if (COMPILER_ID MATCHES "MSVC")
  set(MSVC_${lang}_ARCHITECTURE_ID "${ARCHITECTURE_ID}")
endif()

set(SOURCES_LIST 
    #include
    "./External/include/*/*.hpp"
    "./External/include/*/*.hh"
    "./External/include/*/*.h"
    "./External/include/*.hpp"
    "./External/include/*.hh"
    "./External/include/*.h"
    
    #ray-tracer
    "./Include/Prismarine/*.hpp"
    "./Include/Prismarine/*.inl"
    
    #shaders
    "./ShadersSDK/include/*.*"
    "./ShadersSDK/render/*.*"
    "./ShadersSDK/hlbvh/*.*"
    "./ShadersSDK/tools/*.*"
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
set (LIB_DIR "${DEP_DIR}/External/lib"     "${DEP_DIR}/lib"     "${CMAKE_MODULE_PATH}/lib"     "${DEP_DIR}/External/lib/bullet")
set (INC_DIR "${DEP_DIR}/External/include" "${DEP_DIR}/include" "${CMAKE_MODULE_PATH}/include" "${DEP_DIR}/External/include/bullet" "${PROJECT_SOURCE_DIR}")

set (CMAKE_MODULE_PATH 
    "${DEP_DIR}/modules" 
    "${CMAKE_MODULE_PATH}/modules" 
    "${PROJECT_SOURCE_DIR}/modules/" )

find_package(OpenGL REQUIRED)

set (LIBS 
    ${OPENGL_LIBRARIES}
    glfw3 
    glew32 
    FreeImage
)

set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_EXTENSIONS ON)

if(MSVC)
set (CMAKE_CXX_FLAGS "/EHsc /MD /std:c++latest -DMSVC")
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

add_executable(${APPLICATION_NAME} Source/Examples/Viewer.cpp ${RSOURCES})
target_link_libraries(${APPLICATION_NAME} ${LIBS})

foreach(source IN LISTS RSOURCES)
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach()


set (LIBS 
    ${OPENGL_LIBRARIES}
    glfw3 
    glew 
    FreeImage 
    LinearMath_RelWithDebugInfo
    Bullet3Common_RelWithDebugInfo
    Bullet3Dynamics_RelWithDebugInfo
    Bullet3Collision_RelWithDebugInfo
    BulletCollision_RelWithDebugInfo
    BulletDynamics_RelWithDebugInfo
    BulletInverseDynamics_RelWithDebugInfo
    BulletSoftBody_RelWithDebugInfo
)

set (APPLICATION_NAME prismarine-physics)
add_executable(${APPLICATION_NAME} Source/Examples/Physics.cpp ${RSOURCES})
target_link_libraries(${APPLICATION_NAME} ${LIBS})

foreach(source IN LISTS RSOURCES)
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach()

