#define OS_WIN

#include <iomanip>

#ifdef OS_WIN
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#include <Windows.h>
#endif

#ifdef OS_LNX
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#endif

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "tracer/includes.hpp"
#include "tracer/utils.hpp"
#include "tracer/controller.hpp"
#include "tracer/tracer.hpp"
#include "tracer/intersector.hpp"
#include "tracer/mesh.hpp"
#include "tracer/material.hpp"
#include "tracer/radix.hpp"


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
        glm::vec2 mousepos;
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

        double absscale = 0.75f;
        struct {
            bool show_test_window = false;
            bool show_another_window = false;
            float f = 0.0f;
        } goptions;
    };

    PathTracerApplication::PathTracerApplication(const int32_t& argc, const char ** argv, GLFWwindow * wind) {
        window = wind;

        if (argc < 1) {
            std::cerr << "-m (--model) for load obj model, -s (--scale) for resize model" << std::endl;
        }
        std::string model_input = "";
        
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
            aiProcess_SortByPType
        );
#endif

        rays = new Tracer();
        cam = new Controller();
        cam->setRays(rays);
        supermat = new Material();
        geom = new Mesh();

#ifdef ASSIMP_SUPPORT
        geom->loadMesh(scene->mMeshes, scene->mNumMeshes);
        materialCount = scene->mNumMaterials;
#endif

        glm::mat4 matrix = glm::mat4(1.0f);
        matrix = glm::scale(matrix, glm::vec3(mscale));
        geom->setMaterialOffset(0);
        geom->setTransform(matrix);

        object = new Intersector();
        object->allocate(geom->getNodeCount());
        object->loadMesh(geom);

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

    void PathTracerApplication::mouseMove(const double& x = 0, const double& y = 0) {
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

                material->Get(AI_MATKEY_SHININESS, mat.reflectivity);
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
        //matrix = glm::translate(matrix, glm::vec3(1.0f, 0.0f, 0.0f));
        //matrix = glm::rotate(matrix, float(M_PI) / 4.0f, glm::vec3(1.0f, 0.0f, 0.0f));

        geom->setTransform(matrix);
        object->clearTribuffer();
        object->loadMesh(geom);
        object->build(matrix);

        cam->work(mousepos, diff, lbutton, keys);
        rays->camera(cam->eye, cam->view);

        for (int32_t j = 0;j < depth;j++) {
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

uint32_t rand32() {
    uint32_t x = rand() % 0x100;
    x |= (rand() % 0x100) << 8;
    x |= (rand() % 0x100) << 16;
    x |= (rand() % 0x100) << 24;
    return x;
}




uint64_t ticks(void) {
    const uint64_t ticks_per_second = UINT64_C(10000000);
    static LARGE_INTEGER freq;
    static uint64_t start_time;
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);
    if (!freq.QuadPart) {
        QueryPerformanceFrequency(&freq);
        start_time = value.QuadPart;
    }
    return ((value.QuadPart - start_time) * ticks_per_second) / freq.QuadPart;
}

template<typename F>
uint64_t timed(F && f) {
    auto start = ticks();
    f();
    return ticks() - start;
};

void APIENTRY debug_message(
    GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei, GLchar const * message, void const *) {
    std::cout << std::resetiosflags << std::hex << std::setfill('0')
        << "0x" << std::setw(8) << source << ":"
        << "0x" << std::setw(8) << type << ":"
        << "0x" << std::setw(8) << id << ":"
        << "0x" << std::setw(8) << severity << std::endl
        << message << std::endl;
    if (type == GL_DEBUG_TYPE_ERROR) {
        std::cout << "Press [ENTER] to exit...";
        std::cin.ignore();
        exit(1);
    }
}

#define EACH(i, size) for (auto i = decltype(size)(0); i < size; i++)




int main(const int argc, const char ** argv)
{
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    glfwWindowHint(GLFW_SAMPLES, 0);
    glfwWindowHint(GLFW_RED_BITS, 10);
    glfwWindowHint(GLFW_GREEN_BITS, 10);
    glfwWindowHint(GLFW_BLUE_BITS, 10);
    glfwWindowHint(GLFW_ALPHA_BITS, 2);
    glfwWindowHint(GLFW_DEPTH_BITS, 32); 

    int32_t width = 1024;
    int32_t height = 768;

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    GLFWwindow* window = glfwCreateWindow(width, height, "Simple example", NULL, NULL);
    if (!window) { glfwTerminate(); exit(EXIT_FAILURE); }

#ifdef _WIN32 //Windows DPI scaling
    HWND win = glfwGetWin32Window(window);
    int32_t baseDPI = 96;
    int32_t dpi = baseDPI;//GetDpiForWindow(win);
#else //Other not supported
    int32_t baseDPI = 96;
    int32_t dpi = 96;
#endif
    
    int32_t w = width * ((double)dpi / (double)baseDPI);
    int32_t h = height * ((double)dpi / (double)baseDPI);
    glfwSetWindowSize(window, w, h);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    if (!gladLoadGL()) { glfwTerminate(); exit(EXIT_FAILURE); }

    app = new PaperExample::PathTracerApplication(argc, argv, window);
    app->resizeBuffers(width, height);
    app->resize(w, h);

    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_callback);
    glfwSetCursorPosCallback(window, mouse_move_callback);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        int32_t oldWidth = w;
        int32_t oldHeight = h;
        int32_t oldDPI = dpi;
        
        glfwGetFramebufferSize(window, &w, &h);
        //dpi = GetDpiForWindow(win); // don't rescaling at now (laggy)

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
