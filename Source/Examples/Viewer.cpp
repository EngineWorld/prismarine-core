#include "./Application.hpp"

// application space itself

namespace PrismarineExample {

    class GltfViewer : public PathTracerApplication {
    public:
        GltfViewer(const int32_t& argc, const char ** argv, GLFWwindow * wind) : PathTracerApplication(argc, argv, wind) { execute(argc, argv, wind); };
        GltfViewer() : PathTracerApplication() { };
        void init(const int32_t& argc, const char ** argv);
        void process();
    };

    void GltfViewer::init(const int32_t& argc, const char ** argv) {

        if (argc < 1) std::cerr << "-m (--model) for load obj model, -s (--scale) for resize model" << std::endl;
        std::string model_input = "";
        std::string directory = ".";

        // read arguments
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

        // init material system
        materialManager = new MaterialSet();

        // init ray tracer
        rays = new psm::Pipeline();
        rays->setSkybox(loadCubemap());

        // camera contoller
        cam = new Controller();
        cam->setRays(rays);

#ifdef EXPERIMENTAL_GLTF
        tinygltf::TinyGLTF loader;
        std::string err = "";
        loader.LoadASCIIFromFile(&gltfModel, &err, directory + "/" + model_input);

        // load textures (TODO - native samplers support in ray tracers)
        psm::TextureSet * txset = new psm::TextureSet();
        materialManager->setTextureSet(txset);
        for (int i = 0; i < gltfModel.textures.size(); i++) {
            tinygltf::Texture& gltfTexture = gltfModel.textures[i];
            std::string uri = directory + "/" + gltfModel.images[gltfTexture.source].uri;
            uint32_t rtTexture = txset->loadTexture(uri);
            // todo with rtTexture processing
            rtTextures.push_back(rtTexture);
        }


        // load materials (include PBR)
        materialManager->clearSubmats();
        for (int i = 0; i < gltfModel.materials.size(); i++) {
            tinygltf::Material & material = gltfModel.materials[i];
            psm::VirtualMaterial submat;

            // diffuse?

            int32_t texId = getTextureIndex(material.values["baseColorTexture"].json_double_value);
            submat.diffusePart = texId >= 0 ? rtTextures[texId] : 0;

            if (material.values["baseColorFactor"].number_array.size() >= 3) {
                submat.diffuse = glm::vec4(glm::make_vec3(&material.values["baseColorFactor"].number_array[0]), 1.0f);
            }
            else {
                submat.diffuse = glm::vec4(1.0f);
            }

            // metallic roughness
            texId = getTextureIndex(material.values["metallicRoughnessTexture"].json_double_value);
            submat.specularPart = texId >= 0 ? rtTextures[texId] : 0;
            submat.specular = glm::vec4(1.0f);

            if (material.values["metallicFactor"].number_array.size() >= 1) {
                submat.specular.z = material.values["metallicFactor"].number_array[0];
            }

            if (material.values["roughnessFactor"].number_array.size() >= 1) {
                submat.specular.y = material.values["roughnessFactor"].number_array[0];
            }

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
            materialManager->addSubmat(submat);
        }

        // make raw mesh buffers
        for (int i = 0; i < gltfModel.buffers.size(); i++) {
            GLuint glBuf = -1;
            glCreateBuffers(1, &glBuf);
            glNamedBufferData(glBuf, gltfModel.buffers[i].data.size(), &gltfModel.buffers[i].data.at(0), GL_STATIC_DRAW);
            glBuffers.push_back(glBuf);
        }

        // make buffer views
        BufferViewSet * bfvi = new BufferViewSet();
        for (auto const &bv : gltfModel.bufferViews) {
            VirtualBufferView bfv;
            bfv.offset4 = bv.byteOffset / 4;
            bfv.stride4 = bv.byteStride / 4;
            bfvi->addElement(bfv);
        }

        // load mesh templates (better view objectivity)
        for (int m = 0; m < gltfModel.meshes.size(); m++) {
            std::vector<psm::TriangleArrayInstance *> primitiveVec = std::vector<psm::TriangleArrayInstance *>();

            tinygltf::Mesh &glMesh = gltfModel.meshes[m];
            for (int i = 0; i < glMesh.primitives.size(); i++) {
                tinygltf::Primitive & prim = glMesh.primitives[i];
                psm::TriangleArrayInstance * geom = new psm::TriangleArrayInstance();
                AccessorSet * acs = new AccessorSet();
                geom->setAccessorSet(acs);
                geom->setBufferViewSet(bfvi);

                // make attributes
                std::map<std::string, int>::const_iterator it(prim.attributes.begin());
                std::map<std::string, int>::const_iterator itEnd(prim.attributes.end());

                // load modern mode
                for (auto const &it : prim.attributes) {
                    tinygltf::Accessor &accessor = gltfModel.accessors[it.second];
                    auto& bufferView = gltfModel.bufferViews[accessor.bufferView];

                    // virtual accessor template
                    psm::VirtualAccessor vattr;
                    vattr.offset4 = accessor.byteOffset / 4;
                    vattr.bufferView = accessor.bufferView;

                    // vertex
                    if (it.first.compare("POSITION") == 0) { // vertices
                        vattr.components = 3 - 1;
                        geom->setVertices(glBuffers[bufferView.buffer]);
                        geom->setVertexAccessor(acs->addElement(vattr));
                    }
                    else

                        // normal
                        if (it.first.compare("NORMAL") == 0) {
                            vattr.components = 3 - 1;
                            geom->setNormalAccessor(acs->addElement(vattr));
                        }
                        else

                            // texcoord
                            if (it.first.compare("TEXCOORD_0") == 0) {
                                vattr.components = 2 - 1;
                                geom->setTexcoordAccessor(acs->addElement(vattr));
                            }
                }

                // indices
                // planned of support accessors there
                if (prim.indices >= 0) {
                    tinygltf::Accessor &idcAccessor = gltfModel.accessors[prim.indices];
                    auto& bufferView = gltfModel.bufferViews[idcAccessor.bufferView];
                    geom->setNodeCount(idcAccessor.count / 3);
                    geom->setIndices(glBuffers[bufferView.buffer]);

                    bool isInt16 = idcAccessor.componentType == TINYGLTF_COMPONENT_TYPE_SHORT || idcAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;

                    int32_t loadingOffset = (bufferView.byteOffset + idcAccessor.byteOffset) / (isInt16 ? 2 : 4);
                    geom->setLoadingOffset(loadingOffset);
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

        // create geometry intersector
        intersector = new psm::TriangleHierarchy();
        //intersector->allocate(1024 * 512);
        intersector->allocate(1024 * 2048);

        // init timing state
        time = glfwGetTime() * 1000.f;
        diff = 0;

        /*
        glm::dmat4 matrix(1.0);
        matrix *= glm::scale(glm::dvec3(mscale));
        intersector->clearTribuffer();

#ifdef EXPERIMENTAL_GLTF
        // load meshes
        std::function<void(tinygltf::Node &, glm::dmat4, int)> traverse = [&](tinygltf::Node & node, glm::dmat4 inTransform, int recursive)->void {
            glm::dmat4 localTransform(1.0);
            localTransform *= (node.matrix.size() >= 16 ? glm::make_mat4(node.matrix.data()) : glm::dmat4(1.0));

            localTransform *= (node.translation.size() >= 3 ? glm::translate(glm::make_vec3(node.translation.data())) : glm::dmat4(1.0));
            localTransform *= (node.scale.size() >= 3 ? glm::scale(glm::make_vec3(node.scale.data())) : glm::dmat4(1.0));
            localTransform *= (node.rotation.size() >= 4 ? glm::mat4_cast(glm::make_quat(node.rotation.data())) : glm::dmat4(1.0));

            glm::dmat4 transform = inTransform * localTransform;
            if (node.mesh >= 0) {
                std::vector<psm::TriangleArrayInstance *>& mesh = meshVec[node.mesh]; // load mesh object (it just vector of primitives)
                for (int p = 0; p < mesh.size(); p++) { // load every primitive
                    psm::TriangleArrayInstance * geom = mesh[p];
                    geom->setTransform(transform);
                    intersector->loadMesh(geom);
                }
            }
            else
                if (node.children.size() > 0) {
                    for (int n = 0; n < node.children.size(); n++) {
                        if (recursive >= 0) traverse(gltfModel.nodes[node.children[n]], transform, recursive - 1);
                    }
                }
        };

        // load scene
        uint32_t sceneID = 0;
        if (gltfModel.scenes.size() > 0) {
            for (int n = 0; n < gltfModel.scenes[sceneID].nodes.size(); n++) {
                tinygltf::Node & node = gltfModel.nodes[gltfModel.scenes[sceneID].nodes[n]];
                traverse(node, glm::dmat4(matrix), 2);
            }
        }
#endif*/
    }


    // processing
    void GltfViewer::process() {
        double t = glfwGetTime() * 1000.f;
        diff = t - time;
        time = t;

        // switch to 360 degree view
        cam->work(mousepos, diff, lbutton, keys);
        if (switch360key) {
            rays->switchMode();
            switch360key = false;
        }

        // initial transform
        //glm::dmat4 matrix(1.0);
        //matrix *= glm::scale(glm::dvec3(mscale));

        // clear BVH and load materials
        //intersector->clearTribuffer();









        glm::dmat4 matrix(1.0);
        matrix *= glm::scale(glm::dvec3(mscale));
        intersector->clearTribuffer();

#ifdef EXPERIMENTAL_GLTF
        // load meshes
        std::function<void(tinygltf::Node &, glm::dmat4, int)> traverse = [&](tinygltf::Node & node, glm::dmat4 inTransform, int recursive)->void {
            glm::dmat4 localTransform(1.0);
            localTransform *= (node.matrix.size() >= 16 ? glm::make_mat4(node.matrix.data()) : glm::dmat4(1.0));

            localTransform *= (node.translation.size() >= 3 ? glm::translate(glm::make_vec3(node.translation.data())) : glm::dmat4(1.0));
            localTransform *= (node.scale.size() >= 3 ? glm::scale(glm::make_vec3(node.scale.data())) : glm::dmat4(1.0));
            localTransform *= (node.rotation.size() >= 4 ? glm::mat4_cast(glm::make_quat(node.rotation.data())) : glm::dmat4(1.0));

            glm::dmat4 transform = inTransform * localTransform;
            if (node.mesh >= 0) {
                std::vector<psm::TriangleArrayInstance *>& mesh = meshVec[node.mesh]; // load mesh object (it just vector of primitives)
                for (int p = 0; p < mesh.size(); p++) { // load every primitive
                    psm::TriangleArrayInstance * geom = mesh[p];
                    geom->setTransform(transform);
                    intersector->loadMesh(geom);
                }
            }
            else
                if (node.children.size() > 0) {
                    for (int n = 0; n < node.children.size(); n++) {
                        if (recursive >= 0) traverse(gltfModel.nodes[node.children[n]], transform, recursive - 1);
                    }
                }
        };

        // load scene
        uint32_t sceneID = 0;
        if (gltfModel.scenes.size() > 0) {
            for (int n = 0; n < gltfModel.scenes[sceneID].nodes.size(); n++) {
                tinygltf::Node & node = gltfModel.nodes[gltfModel.scenes[sceneID].nodes[n]];
                traverse(node, glm::dmat4(matrix), 2);
            }
        }
#endif









        materialManager->loadToVGA();

        // build BVH in device
        intersector->markDirty();
        intersector->build();

        // process ray tracing
        rays->camera(cam->eye, cam->view);
        for (int32_t j = 0; j < depth; j++) {
            if (rays->getRayCount() <= 0) break;
            rays->intersection(intersector);
            rays->applyMaterials(materialManager); // low level function for getting surface materials (may have few materials)
            rays->shade(); // low level function for change rays
            rays->reclaim();
        }
        rays->sample();
        rays->render();

        glFinish();
    }

};






// main (don't change)
//////////////////////

PrismarineExample::GltfViewer * app = nullptr;

static void error_callback(int32_t error, const char* description) {
    std::cerr << ("Error: \n" + std::string(description)) << std::endl;
}

static void key_callback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (action == GLFW_PRESS) app->passKeyDown(key);
    if (action == GLFW_RELEASE) app->passKeyRelease(key);
}

static void mouse_callback(GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
    if (action == GLFW_PRESS) app->mousePress(button);
    if (action == GLFW_RELEASE) app->mouseRelease(button);
}

static void mouse_move_callback(GLFWwindow* window, double x, double y) {
    app->mouseMove(x, y);
}

int main(const int argc, const char ** argv)
{
    if (!glfwInit()) exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
#ifdef USE_OPENGL_45_COMPATIBLE
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
#endif

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_NO_ERROR, GLFW_TRUE);


    // planks 
    // great around 30 (or higher) FPS
    // fine around 15 FPS
    // "may works" around 5-10 FPS

    // GPU rating (by increment)
    // GTX 1060 
    // GTX 1070
    // VEGA 56
    // VEGA 64
    // GTX 1080 Ti
    // Titan Xp

    // VEGA 64 should work fine without interlacing, for VEGA 56 will harder, GTX 1080 Ti should work great, GTX 1070 should also work great with interlacing
    int32_t baseWidth = 640;
    int32_t baseHeight = 360;

    // GTX 1070 should work fine without interlacing, VEGA 64 should work great, GTX 1060 with interlacing also should work great
    //int32_t baseWidth = 400;
    //int32_t baseHeight = 300;

    // VEGA 64 (with interlacing) should work fine, GTX 1080 Ti may works without interlacing
    //int32_t baseWidth = 960;
    //int32_t baseHeight = 540;

    // VEGA 56 test with or without interlacing (or GTX 1070 with interlacing), from VEGA 56 with interlacing should work fine, GTX 1080 Ti should work fine without interlacing in some cases
    //int32_t baseWidth = 800;
    //int32_t baseHeight = 450;

    // GTX 1080 Ti should work fine with interlacing (1440p)
    //int32_t baseWidth = 1280;
    //int32_t baseHeight = 640;

    GLFWwindow* window = glfwCreateWindow(baseWidth, baseHeight, "Simple example", NULL, NULL);
    if (!window) { glfwTerminate(); exit(EXIT_FAILURE); }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    if (glewInit() != GLEW_OK) glfwTerminate();

    app = new PrismarineExample::GltfViewer();
    glfwSetErrorCallback(error_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_callback);
    glfwSetCursorPosCallback(window, mouse_move_callback);
    app->execute(argc, argv, window);

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
