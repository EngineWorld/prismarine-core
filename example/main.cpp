﻿#define OS_WIN

#include <iomanip>

#ifdef OS_WIN
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
//#include <Windows.h>
#endif

#ifdef OS_LNX
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#endif

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <tiny_gltf.h>

#include "tracer/includes.hpp"
#include "tracer/utils.hpp"
#include "tracer/controller.hpp"
#include "tracer/tracer.hpp"
#include "tracer/intersector.hpp"
#include "tracer/mesh.hpp"
#include "tracer/material.hpp"
#include "tracer/radix.hpp"
#include <functional>

namespace PaperExample {
    using namespace Paper;

    class PathTracerApplication {
    public:
        PathTracerApplication(const int32_t& argc, const char ** argv, GLFWwindow * wind);
        void passKeyDown(const int32_t& key);
        void passKeyRelease(const int32_t& key);
        void mousePress(const int32_t& button);
        void mouseRelease(const int32_t& button);
        void mouseMove(const double& x, const double& y);
        void draw();
        void proccessUI();
        void resize(const int32_t& width, const int32_t& height);
        void resizeBuffers(const int32_t& width, const int32_t& height);

        double dpiscaling = 1.0f;

    private:
        
        GLFWwindow * window;
        Tracer * rays;
        Intersector * object;
        Mesh * geom;
        Controller * cam;
        Material * supermat;
        
        double time = 0;
        double diff = 0;
        glm::dvec2 mousepos;
        double mscale = 1.0f;
        int32_t depth = 16;
        int32_t switch360key = false;
        bool lbutton = false;
        bool keys[10] = { false , false , false , false , false , false , false, false, false };
        
#ifdef TOL_SUPPORT
        std::vector<tinyobj::material_t> materials;
#endif

#ifdef ASSIMP_SUPPORT
        aiMaterial ** materials;
        int32_t materialCount;

        Assimp::Importer importer;
        const aiScene* scene;
#endif

#ifdef EXPERIMENTAL_GLTF
        tinygltf::Model gltfModel;
        //std::vector<Mesh *> primitiveVec = std::vector<Mesh *>();
        std::vector<std::vector<Mesh *>> meshVec = std::vector<std::vector<Mesh *>>();
        std::vector<GLuint> glBuffers = std::vector<GLuint>();
        std::vector<uint32_t> rtTextures = std::vector<uint32_t>();
#endif

        //double absscale = 0.75f;
        struct {
            bool show_test_window = false;
            bool show_another_window = false;
            float f = 0.0f;
        } goptions;
    };


    uint32_t _byType(int &type) {
        switch (type) {
            case TINYGLTF_TYPE_VEC4:
            return 4;
            break;

            case TINYGLTF_TYPE_VEC3:
            return 3;
            break;

            case TINYGLTF_TYPE_VEC2:
            return 2;
            break;

            case TINYGLTF_TYPE_SCALAR:
            return 1;
            break;
        }
        return 1;
    }


    int32_t getTextureIndex(std::map<std::string, double> &mapped) {
        return mapped.count("index") > 0 ? mapped["index"] : -1;
        //return mapped["index"];
    }


    PathTracerApplication::PathTracerApplication(const int32_t& argc, const char ** argv, GLFWwindow * wind) {
        window = wind;

        if (argc < 1) {
            std::cerr << "-m (--model) for load obj model, -s (--scale) for resize model" << std::endl;
        }
        std::string model_input = "";
        std::string directory = ".";

        for (int i = 1; i < argc; ++i) {
            std::string arg = std::string(argv[i]);
            if ((arg == "-m") || (arg == "--model")) {
                if (i + 1 < argc) {
                    model_input = std::string(argv[++i]);
                }
                else {
                    std::cerr << "Model filename required" << std::endl;
                }
            }
            else
            if ((arg == "-s") || (arg == "--scale")) {
                    if (i + 1 < argc) {
                        mscale = std::stof(argv[++i]);
                    }
            }
            else
            if ((arg == "-di") || (arg == "--dir")) {
                if (i + 1 < argc) {
                    directory = std::string(argv[++i]);
                }
            }
            else
            if ((arg == "-d") || (arg == "--depth")) {
                    if (i + 1 < argc) {
                        depth = std::stoi(argv[++i]);
                    }
            }
        }

        if (model_input == "") {
            std::cerr << "No model found :(" << std::endl;
        }

#ifdef TOL_SUPPORT
        std::vector<tinyobj::shape_t> shapes;
        tinyobj::attrib_t attrib;
        std::string err;
        tinyobj::LoadObj(&attrib, &shapes, &materials, &err, model_input.c_str());
#endif

#ifdef ASSIMP_SUPPORT
        scene = importer.ReadFile(model_input, 0 | 
            aiProcess_CalcTangentSpace |
            aiProcess_Triangulate |
            aiProcess_JoinIdenticalVertices |
            aiProcess_SortByPType | 
            aiProcess_TransformUVCoords
        );
#endif

        rays = new Tracer();
        cam = new Controller();
        cam->setRays(rays);
        supermat = new Material();
        geom = new Mesh();

#ifdef EXPERIMENTAL_GLTF
        tinygltf::TinyGLTF loader;
        std::string err = "";
        loader.LoadASCIIFromFile(&gltfModel, &err, directory + "/" + model_input);

        // load textures (TODO - native samplers support in ray tracers)
        for (int i = 0; i < gltfModel.textures.size(); i++) {
            tinygltf::Texture& gltfTexture = gltfModel.textures[i];
            std::string uri = directory + "/" + gltfModel.images[gltfTexture.source].uri;
            uint32_t rtTexture = supermat->loadTexture(uri);
            // todo with rtTexture processing
            rtTextures.push_back(rtTexture);
        }

        // load materials (include PBR)
        supermat->clearSubmats();
        for (int i = 0; i < gltfModel.materials.size(); i++) {
            tinygltf::Material & material = gltfModel.materials[i];
            Paper::Material::Submat submat;

            // diffuse?
            int32_t texId = getTextureIndex(material.values["baseColorTexture"].json_double_value);
            submat.diffusePart = texId >= 0 ? rtTextures[texId] : 0;
            submat.diffuse = glm::vec4(1.0f);

            // metallic roughness
            texId = getTextureIndex(material.values["metallicRoughnessTexture"].json_double_value);
            submat.specularPart = texId >= 0 ? rtTextures[texId] : 0;
            submat.specular = glm::vec4(0.0f);

            // emission
            if (material.additionalValues["emissiveFactor"].number_array.size() >= 3) {
                submat.emissive = glm::vec4(glm::make_vec3(&material.additionalValues["emissiveFactor"].number_array[0]), 1.0f);
            }
            else {
                submat.emissive = glm::vec4(0.0f);
            }
            
            // emissive texture
            texId = getTextureIndex(material.additionalValues["emissiveTexture"].json_double_value);
            submat.emissivePart = texId >= 0 ? rtTextures[texId] : 0;

            // normal map
            texId = getTextureIndex(material.additionalValues["normalTexture"].json_double_value);
            submat.bumpPart = texId >= 0 ? rtTextures[texId] : 0;

            // load material
            supermat->addSubmat(submat);
        }

        // make raw buffers
        for (int i = 0; i < gltfModel.buffers.size();i++) {
            GLuint glBuf = -1;
            glCreateBuffers(1, &glBuf);
            glNamedBufferData(glBuf, gltfModel.buffers[i].data.size(), &gltfModel.buffers[i].data.at(0), GL_STATIC_DRAW);
            glBuffers.push_back(glBuf);
        }

        // make buffers for OpenGL/RayTracers
        /*
        for (int i = 0; i < gltfModel.bufferViews.size();i++) {
            tinygltf::BufferView &bview = gltfModel.bufferViews[i];

            GLuint glBuf = -1;
            glCreateBuffers(1, &glBuf);
            glNamedBufferData(glBuf, bview.byteLength, &gltfModel.buffers[bview.buffer].data.at(bview.byteOffset), GL_STATIC_DRAW);

            glBuffers.push_back(glBuf);
        }
        */

        // load meshes (better view objectivity)
        for (int m = 0; m < gltfModel.meshes.size();m++) {
        //int m = 0; {
            std::vector<Mesh *> primitiveVec = std::vector<Mesh *>();

            tinygltf::Mesh &glMesh = gltfModel.meshes[m];
            for (int i = 0; i < glMesh.primitives.size();i++) {
                tinygltf::Primitive & prim = glMesh.primitives[i];
                Mesh * geom = new Mesh();

                // make attributes
                std::map<std::string, int>::const_iterator it(prim.attributes.begin());
                std::map<std::string, int>::const_iterator itEnd(prim.attributes.end());

                geom->attributeUniformData.haveTexcoord = false;
                geom->attributeUniformData.haveNormal = false;
                geom->attributeUniformData.mode = 0;
                geom->attributeUniformData.stride = 0;

                // load modern mode
                for (; it != itEnd; it++) {
                    tinygltf::Accessor &accessor = gltfModel.accessors[it->second];
                    auto& bufferView = gltfModel.bufferViews[accessor.bufferView];

                    if (it->first.compare("POSITION") == 0) { // vertices
                        geom->attributeUniformData.vertexOffset = (accessor.byteOffset + bufferView.byteOffset) / 4;
                        geom->attributeUniformData.stride = (bufferView.byteStride <= 4) ? 0 : (bufferView.byteStride / 4);
                        geom->setVertices(glBuffers[bufferView.buffer]);
                    } else
                    
                    if (it->first.compare("NORMAL") == 0) {
                        geom->attributeUniformData.haveNormal = true;
                        geom->attributeUniformData.normalOffset = (accessor.byteOffset + bufferView.byteOffset) / 4;
                    } else

                    if (it->first.compare("TEXCOORD_0") == 0) {
                        geom->attributeUniformData.haveTexcoord = true;
                        geom->attributeUniformData.texcoordOffset = (accessor.byteOffset + bufferView.byteOffset) / 4;
                    }
                }

                // indices
                if (prim.indices >= 0) {
                    tinygltf::Accessor &idcAccessor = gltfModel.accessors[prim.indices];
                    auto& bufferView = gltfModel.bufferViews[idcAccessor.bufferView];
                    geom->setNodeCount(idcAccessor.count / 3);
                    geom->setIndices(glBuffers[bufferView.buffer]);

                    bool isInt16 = idcAccessor.componentType == TINYGLTF_COMPONENT_TYPE_SHORT || idcAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
                    geom->setLoadingOffset((bufferView.byteOffset + idcAccessor.byteOffset) / (isInt16 ? 2 : 4));
                    geom->setIndexed(true);

                    // is 16-bit indices?
                    geom->useIndex16bit(isInt16);
                }

                // use material
                geom->setMaterialOffset(prim.material);

                // if triangles, then load mesh
                if (prim.mode == TINYGLTF_MODE_TRIANGLES) {
                    primitiveVec.push_back(geom);
                }
            }

            meshVec.push_back(primitiveVec);
        }
#endif



#ifdef ASSIMP_SUPPORT
        geom->loadMesh(scene->mMeshes, scene->mNumMeshes);
        materialCount = scene->mNumMaterials;
#endif

        glm::mat4 matrix = glm::mat4(1.0f);
        matrix = glm::scale(matrix, glm::vec3(mscale));
        geom->setMaterialOffset(0);
        geom->setTransform(matrix);

#ifdef ASSIMP_SUPPORT
        //object->allocate(geom->getNodeCount());
        //object->loadMesh(geom);
#else
        object = new Intersector();
        object->allocate(1024 * 1024);
#endif

        time = milliseconds();
        diff = 0;
    }

    void PathTracerApplication::passKeyDown(const int32_t& key) {
        if (key == GLFW_KEY_W) keys[kW] = true;
        if (key == GLFW_KEY_A) keys[kA] = true;
        if (key == GLFW_KEY_S) keys[kS] = true;
        if (key == GLFW_KEY_D) keys[kD] = true;
        if (key == GLFW_KEY_Q) keys[kQ] = true;
        if (key == GLFW_KEY_E) keys[kE] = true;
        if (key == GLFW_KEY_C) keys[kC] = true;
        if (key == GLFW_KEY_SPACE) keys[kSpc] = true;
        if (key == GLFW_KEY_LEFT_SHIFT) keys[kSft] = true;
        if (key == GLFW_KEY_K) keys[kK] = true;
    }

    void PathTracerApplication::passKeyRelease(const int32_t& key) {
        if (key == GLFW_KEY_W) keys[kW] = false;
        if (key == GLFW_KEY_A) keys[kA] = false;
        if (key == GLFW_KEY_S) keys[kS] = false;
        if (key == GLFW_KEY_D) keys[kD] = false;
        if (key == GLFW_KEY_Q) keys[kQ] = false;
        if (key == GLFW_KEY_E) keys[kE] = false;
        if (key == GLFW_KEY_C) keys[kC] = false;
        if (key == GLFW_KEY_SPACE) keys[kSpc] = false;
        if (key == GLFW_KEY_LEFT_SHIFT) keys[kSft] = false;
        if (key == GLFW_KEY_K) {
            if (keys[kK]) switch360key = true;
            keys[kK] = false;
        }
    }

    void PathTracerApplication::mousePress(const int32_t& button) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) lbutton = true;
    }

    void PathTracerApplication::mouseRelease(const int32_t& button) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) lbutton = false;
    }

    void PathTracerApplication::mouseMove(const double& x, const double& y) {
        mousepos.x = x;
        mousepos.y = y;
    }

    void PathTracerApplication::resizeBuffers(const int32_t& width, const int32_t& height) {
        rays->resizeBuffers(width, height);
    }

    void PathTracerApplication::resize(const int32_t& width, const int32_t& height) {
        rays->resize(width, height);
    }

    void PathTracerApplication::proccessUI() {
#ifdef ENABLE_GUI
        double dscale = dpiscaling * absscale;
        ImGui::SetNextWindowSize(ImVec2(400 * dscale, 200 * dscale), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("Testing");
        ImGui::SetWindowFontScale(dscale);
            
        ImGui::Text("Hello, world!");
        ImGui::SliderFloat("float", &goptions.f, 0.0f, 1.0f);
        ImGui::ColorEdit3("clear color", (float*)&goptions.clear_color);
        if (ImGui::Button("Test Window")) goptions.show_test_window = goptions.show_test_window ? 0 : 1;
        if (ImGui::Button("Another Window")) goptions.show_another_window = goptions.show_another_window ? 0 : 1;
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
#endif

    }

    void PathTracerApplication::draw() {
        double t = milliseconds();
        diff = t - time;
        time = t;

        this->proccessUI();
        if (switch360key) {
            rays->switchMode();
            switch360key = false;
        }

#ifdef TOL_SUPPORT
        supermat->submats.resize(0);

        for (int32_t i = 0; i < materials.size(); i++) {
            Material::Submat mat;
            mat.reflectivity = materials[i].shininess;
            mat.diffusePart = supermat->loadTexture(materials[i].diffuse_texname);
            mat.emissivePart = supermat->loadTexture(materials[i].emissive_texname);
            mat.specularPart = supermat->loadTexture(materials[i].specular_texname);
            mat.bumpPart = supermat->loadTexture(materials[i].normal_texname != "" ? materials[i].normal_texname : materials[i].bump_texname);

            mat.diffuse = glm::vec4(materials[i].diffuse[0], materials[i].diffuse[1], materials[i].diffuse[2], 1.0f);
            mat.specular = glm::vec4(materials[i].specular[0], materials[i].specular[1], materials[i].specular[2], 1.0f);
            mat.emissive = glm::vec4(materials[i].emission[0], materials[i].emission[1], materials[i].emission[2], 1.0f);
            mat.transmission = glm::vec4(materials[i].transmittance[0], materials[i].transmittance[1], materials[i].transmittance[2], 1.0f);
            mat.ior = materials[i].ior;
            mat.flags |= (1 << 6);

            supermat->submats.resize( std::max(size_t(i + 1), supermat->submats.size()));
            supermat->submats[i] = mat;
        }

        supermat->loadToVGA();
#endif

#ifdef ASSIMP_SUPPORT
        supermat->submats.resize(0);

        if (scene->HasMaterials()) {
            for (int32_t i = 0; i < materialCount; i++) {
                Material::Submat mat;
                const aiMaterial * material = scene->mMaterials[i];

                float coef = 0.0f;
                aiColor4D dcolor;
                material->Get(AI_MATKEY_COLOR_DIFFUSE, dcolor);
                mat.diffuse = { dcolor.r, dcolor.g, dcolor.b, dcolor.a };

                material->Get(AI_MATKEY_COLOR_EMISSIVE, dcolor);
                mat.emissive = { dcolor.r, dcolor.g, dcolor.b, dcolor.a };

                material->Get(AI_MATKEY_COLOR_SPECULAR, dcolor);
                mat.specular = { dcolor.r, dcolor.g, dcolor.b, dcolor.a };

                material->Get(AI_MATKEY_COLOR_TRANSPARENT, dcolor);
                mat.transmission = { dcolor.r, dcolor.g, dcolor.b, dcolor.a };

                material->Get(AI_MATKEY_SHININESS_STRENGTH, mat.roughness);
                material->Get(AI_MATKEY_REFRACTI, mat.ior);
                mat.flags |= (1 << 6);


                // Load textrues by lodepng (later will support with ASSIMP, or another one)
                aiString tsc;

                tsc = "";
                material->GetTexture(aiTextureType_DIFFUSE, 0, &tsc);
                mat.diffusePart = supermat->loadTexture(std::string(tsc.data, tsc.length));

                tsc = "";
                material->GetTexture(aiTextureType_EMISSIVE, 0, &tsc);
                mat.emissivePart = supermat->loadTexture(std::string(tsc.data, tsc.length));

                tsc = "";
                material->GetTexture(aiTextureType_SPECULAR, 0, &tsc);
                mat.specularPart = supermat->loadTexture(std::string(tsc.data, tsc.length));

                tsc = "";
                material->GetTexture(aiTextureType_NORMALS, 0, &tsc);
                mat.bumpPart = supermat->loadTexture(std::string(tsc.data, tsc.length));

                if (tsc.length == 0) {
                    material->GetTexture(aiTextureType_HEIGHT, 0, &tsc);
                    mat.bumpPart = supermat->loadTexture(std::string(tsc.data, tsc.length));
                }

                supermat->submats.resize(std::max(size_t(i + 1), supermat->submats.size()));
                supermat->submats[i] = mat;
            }
        }

        supermat->loadToVGA();
#endif

        glm::mat4 matrix(1.0f);
        matrix = glm::scale(matrix, glm::vec3(mscale));
        //matrix = glm::rotate(matrix, float(t) / 10000.f, glm::vec3(1.0f, 0.0f, 0.0f));
        //matrix = glm::rotate(matrix, float(t) / 20000.f, glm::vec3(0.0f, 0.0f, 1.0f));

        //matrix = glm::translate(matrix, glm::vec3(1.0f, 0.0f, 0.0f));
        //matrix = glm::rotate(matrix, float(M_PI) / 4.0f, glm::vec3(1.0f, 0.0f, 0.0f));

        // load meshes to BVH
        object->clearTribuffer();
#ifdef EXPERIMENTAL_GLTF
        supermat->loadToVGA();

        /*
        for (int g = 0; g < meshVec.size();g++) {
            std::vector<Paper::Mesh *>& mesh = meshVec[g];
            for (int p = 0; p < mesh.size();p++) {
                Paper::Mesh * geom = mesh[p];



                geom->setTransform(matrix);
                object->loadMesh(geom);

            }
        }
        */

        // load tree
        std::function<void(tinygltf::Node &, glm::dmat4, int)> traverse = [&](tinygltf::Node & node, glm::dmat4 inTransform, int recursive)->void {
            glm::dmat4 ltransform = glm::dmat4(1.0);
            ltransform *= (node.matrix.size() >= 16 ? glm::make_mat4(node.matrix.data()) : glm::dmat4(1.0));
            
            
            ltransform *= (node.translation.size() >= 3 ? glm::translate(glm::dmat4(1.0), glm::make_vec3(node.translation.data())) : glm::dmat4(1.0));
            ltransform *= (node.scale.size() >= 3 ? glm::scale(glm::dmat4(1.0), glm::make_vec3(node.scale.data())) : glm::dmat4(1.0));
            ltransform *= (node.rotation.size() >= 4 ? glm::mat4_cast(glm::make_quat(node.rotation.data())) : glm::dmat4(1.0));
            //ltransform *= (node.rotation.size() >= 4 ? glm::rotate(glm::dmat4(1.0), node.rotation[0], glm::make_vec3(node.rotation.data() + 1)) : glm::dmat4(1.0));

            glm::dmat4 transform = inTransform * ltransform;

            if (node.mesh >= 0) {
                std::vector<Paper::Mesh *>& mesh = meshVec[node.mesh]; // load mesh object (it just vector of primitives)
                for (int p = 0; p < mesh.size(); p++) { // load every primitive
                    Paper::Mesh * geom = mesh[p];
                    geom->setTransform(transform);
                    object->loadMesh(geom);
                }
            }
            else
            if (node.children.size() > 0) {
                for (int n = 0; n < node.children.size(); n++) {
                    if (recursive >= 0) traverse(gltfModel.nodes[node.children[n]], transform, recursive-1);
                }
            }
        };

        // load scene
        uint32_t sceneID = 0;
        for (int n = 0; n < gltfModel.scenes[sceneID].nodes.size();n++) {
            tinygltf::Node & node = gltfModel.nodes[gltfModel.scenes[sceneID].nodes[n]];
            traverse(node, glm::dmat4(matrix), 2);
        }

#else 
        geom->setTransform(matrix);
        object->loadMesh(geom);
#endif

        object->build(matrix);

        cam->work(mousepos, diff, lbutton, keys);
        rays->camera(cam->eye, cam->view);

        for (int32_t j = 0;j < depth;j++) {
        //for (int32_t j = 0; j < 1; j++) {
            if (rays->getRayCount() <= 0) break;
            rays->resetHits();
            rays->intersection(object);
            rays->shade(supermat);
            rays->reclaim();
        }

        rays->sample();
        rays->render();
    }
}

PaperExample::PathTracerApplication * app;

static void error_callback(int32_t error, const char* description){
    fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods){
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (action == GLFW_PRESS) app->passKeyDown(key);
    if (action == GLFW_RELEASE) app->passKeyRelease(key);
}

static void mouse_callback(GLFWwindow* window, int32_t button, int32_t action, int32_t mods){
    if (action == GLFW_PRESS) app->mousePress(button);
    if (action == GLFW_RELEASE) app->mouseRelease(button);
}

static void mouse_move_callback(GLFWwindow* window, double x, double y){
    app->mouseMove(x, y);
}

const unsigned super_sampling = 2;

int main(const int argc, const char ** argv)
{
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int32_t width = 640;
    int32_t height = 360;

    GLFWwindow* window = glfwCreateWindow(width, height, "Simple example", NULL, NULL);
    if (!window) { glfwTerminate(); exit(EXIT_FAILURE); }

#ifdef _WIN32 //Windows DPI scaling
    HWND win = glfwGetWin32Window(window);
    int32_t baseDPI = 96;
    int32_t dpi = baseDPI;
#else //Other not supported
    int32_t baseDPI = 96;
    int32_t dpi = 96;
#endif
    
    // DPI scaling for Windows
#ifdef MSVC and _WIN32
    dpi = GetDpiForWindow(win);
#endif

    int32_t w = width * ((double)dpi / (double)baseDPI);
    int32_t h = height * ((double)dpi / (double)baseDPI);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    if (!gladLoadGL()) { glfwTerminate(); exit(EXIT_FAILURE); }

    app = new PaperExample::PathTracerApplication(argc, argv, window);
    app->resizeBuffers(width * super_sampling, height * super_sampling);
    app->resize(w, h);

    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_callback);
    glfwSetCursorPosCallback(window, mouse_move_callback);

    glfwSetWindowSize(window, w, h);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        int32_t oldWidth = w;
        int32_t oldHeight = h;
        int32_t oldDPI = dpi;
        
        glfwGetFramebufferSize(window, &w, &h);

        // DPI scaling for Windows
#ifdef MSVC and _WIN32
        dpi = GetDpiForWindow(win);
#endif

        double ratio = ((double)dpi / (double)baseDPI);
        if (oldDPI != dpi) {
            w = width  * ratio;
            h = height * ratio;
            glfwSetWindowSize(window, w, h);
        }

        if (oldWidth != w || oldHeight != h) {
            width  = w / ratio;
            height = h / ratio;
            app->resize(w, h);
        }

        app->dpiscaling = ratio;
        app->draw();

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
