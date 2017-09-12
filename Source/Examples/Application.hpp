#define OS_WIN

#include <iomanip>

#ifdef OS_WIN
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#endif

#ifdef OS_LNX
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#endif

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <tiny_gltf.h>

#include "Prismarine/Prismarine.hpp"
#include <functional>

namespace PrismarineExample {

    using namespace psm;

	const std::string bgTexName = "background.jpg";

	GLuint loadCubemap() {
		FREE_IMAGE_FORMAT formato = FreeImage_GetFileType(bgTexName.c_str(), 0);
		if (formato == FIF_UNKNOWN) {
			return 0;
		}
		FIBITMAP* imagen = FreeImage_Load(formato, bgTexName.c_str());
		if (!imagen) {
			return 0;
		}

		FIBITMAP* temp = FreeImage_ConvertTo32Bits(imagen);
		FreeImage_Unload(imagen);
		imagen = temp;

		uint32_t width = FreeImage_GetWidth(imagen);
		uint32_t height = FreeImage_GetHeight(imagen);

		GLuint texture = 0;
		glCreateTextures(GL_TEXTURE_2D, 1, &texture);
		glTextureStorage2D(texture, 1, GL_RGBA8, width, height);
		glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		uint8_t * pixelsPtr = FreeImage_GetBits(imagen);
		glTextureSubImage2D(texture, 0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, pixelsPtr);

		return texture;
	}

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
	}

	const int32_t kW = 0;
	const int32_t kA = 1;
	const int32_t kS = 2;
	const int32_t kD = 3;
	const int32_t kQ = 4;
	const int32_t kE = 5;
	const int32_t kSpc = 6;
	const int32_t kSft = 7;
	const int32_t kC = 8;
	const int32_t kK = 9;
	const int32_t kM = 10;
	const int32_t kL = 11;

    class Controller {
        bool monteCarlo = true;

    public:
        glm::dvec3 eye = glm::dvec3(0.0f, 6.0f, 6.0f);
        glm::dvec3 view = glm::dvec3(0.0f, 2.0f, 0.0f);
        glm::dvec2 mposition;
        psm::Pipeline * raysp;

        glm::dmat4 project() {
#ifdef USE_CAD_SYSTEM
            return glm::lookAt(eye, view, glm::dvec3(0.0f, 0.0f, 1.0f));
#elif USE_180_SYSTEM
            return glm::lookAt(eye, view, glm::dvec3(0.0f, -1.0f, 0.0f));
#else
            return glm::lookAt(eye, view, glm::dvec3(0.0f, 1.0f, 0.0f));
#endif
        }

        void setRays(psm::Pipeline * r) {
            raysp = r;
        }

        void work(const glm::dvec2 &position, const double &diff, const bool &mouseleft, const bool keys[10]) {
            glm::dmat4 viewm = project();
            glm::dmat4 unviewm = glm::inverse(viewm);
            glm::dvec3 ca = (viewm * glm::dvec4(eye, 1.0f)).xyz();
            glm::dvec3 vi = (viewm * glm::dvec4(view, 1.0f)).xyz();

            bool isFocus = true;

            if (mouseleft && isFocus)
            {
                glm::dvec2 mpos = glm::dvec2(position) - mposition;
                double diffX = mpos.x;
                double diffY = mpos.y;
                if (glm::abs(diffX) > 0.0) this->rotateX(vi, diffX);
                if (glm::abs(diffY) > 0.0) this->rotateY(vi, diffY);
                if (monteCarlo) raysp->clearSampler();
            }
            mposition = glm::dvec2(position);

            if (keys[kW] && isFocus)
            {
                this->forwardBackward(ca, vi, diff);
                if (monteCarlo) raysp->clearSampler();
            }

            if (keys[kS] && isFocus)
            {
                this->forwardBackward(ca, vi, -diff);
                if (monteCarlo) raysp->clearSampler();
            }

            if (keys[kA] && isFocus)
            {
                this->leftRight(ca, vi, diff);
                if (monteCarlo) raysp->clearSampler();
            }

            if (keys[kD] && isFocus)
            {
                this->leftRight(ca, vi, -diff);
                if (monteCarlo) raysp->clearSampler();
            }

            if ((keys[kE] || keys[kSpc]) && isFocus)
            {
                this->topBottom(ca, vi, diff);
                if (monteCarlo) raysp->clearSampler();
            }

            if ((keys[kQ] || keys[kSft] || keys[kC]) && isFocus)
            {
                this->topBottom(ca, vi, -diff);
                if (monteCarlo) raysp->clearSampler();
            }

            eye = (unviewm * glm::vec4(ca, 1.0f)).xyz();
            view = (unviewm * glm::vec4(vi, 1.0f)).xyz();
        }

        void leftRight(glm::dvec3 &ca, glm::dvec3 &vi, const double &diff) {
            ca.x -= diff / 100.0f;
            vi.x -= diff / 100.0f;
        }
        void topBottom(glm::dvec3 &ca, glm::dvec3 &vi, const double &diff) {
            ca.y += diff / 100.0f;
            vi.y += diff / 100.0f;
        }
        void forwardBackward(glm::dvec3 &ca, glm::dvec3 &vi, const double &diff) {
            ca.z -= diff / 100.0f;
            vi.z -= diff / 100.0f;
        }
        void rotateY(glm::dvec3 &vi, const double &diff) {
            glm::dmat4 rot = glm::rotate(-diff / float(raysp->displayHeight) / 0.5f, glm::dvec3(1.0f, 0.0f, 0.0f));
            vi = (rot * glm::dvec4(vi, 1.0f)).xyz();
        }
        void rotateX(glm::dvec3 &vi, const double &diff) {
            glm::dmat4 rot = glm::rotate(-diff / float(raysp->displayHeight) / 0.5f, glm::dvec3(0.0f, 1.0f, 0.0f));
            vi = (rot * glm::dvec4(vi, 1.0f)).xyz();
        }
    };

    class PathTracerApplication {
    public:
		PathTracerApplication(const int32_t& argc, const char ** argv, GLFWwindow * wind) { };
		PathTracerApplication() { };

        void passKeyDown(const int32_t& key);
        void passKeyRelease(const int32_t& key);
        void mousePress(const int32_t& button);
        void mouseRelease(const int32_t& button);
        void mouseMove(const double& x, const double& y);
		
        void resize(const int32_t& width, const int32_t& height);
        void resizeBuffers(const int32_t& width, const int32_t& height);
		void saveHdr(std::string name = "");

		virtual void init(const int32_t& argc, const char ** argv) = 0;
		virtual void process() = 0;
		virtual void execute(const int32_t& argc, const char ** argv, GLFWwindow * wind);

    protected:
		const double measureSeconds = 2.0;
		const double superSampling = 2.0; // IT SHOULD! Should be double resolution!

		int32_t baseWidth = 1;
		int32_t baseHeight = 1;

        GLFWwindow * window;
        psm::Pipeline * rays;
        psm::TriangleHierarchy * intersector;
        Controller * cam;
        psm::MaterialSet * materialManager;
        
        double time = 0;
        double diff = 0;
        glm::dvec2 mousepos;
        double mscale = 1.0f;
        int32_t depth = 16;
        int32_t switch360key = false;
        int32_t img_counter = 0;
        bool lbutton = false;
        bool keys[12] = { false , false , false , false , false , false , false, false, false, false, false };

#ifdef EXPERIMENTAL_GLTF
        tinygltf::Model gltfModel;
        std::vector<std::vector<psm::TriangleArrayInstance *>> meshVec = std::vector<std::vector<psm::TriangleArrayInstance *>>();
        std::vector<GLuint> glBuffers = std::vector<GLuint>();
        std::vector<uint32_t> rtTextures = std::vector<uint32_t>();
#endif

        struct {
            bool show_test_window = false;
            bool show_another_window = false;
            float f = 0.0f;
        } goptions;
    };

	void PathTracerApplication::execute(const int32_t& argc, const char ** argv, GLFWwindow * wind) {
		window = wind;
		glfwGetWindowSize(window, &baseWidth, &baseHeight);

		// get DPI scaling
		float scale = 1.0f;
		glfwGetWindowContentScale(window, &scale, nullptr);

		// make DPI scaled
		int32_t windowWidth = baseWidth * scale, windowHeight = baseHeight * scale;
		glfwSetWindowSize(window, windowWidth, windowHeight);

		// get canvas size
		int32_t canvasWidth = baseWidth, canvasHeight = baseHeight;
		glfwGetFramebufferSize(window, &canvasWidth, &canvasHeight);

		// init or prerender data
		this->init(argc, argv);

		// resize buffers and canvas
		this->resizeBuffers(int(double(baseWidth) * double(superSampling)), int(double(baseHeight) * double(superSampling)));
		this->resize(canvasWidth, canvasHeight);

		// rendering itself
		double lastTime = glfwGetTime();
		double prevFrameTime = lastTime;
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();

			int32_t oldWidth = windowWidth, oldHeight = windowHeight;
			float oldScale = scale;

			// DPI scaling for Windows
			{
				glfwGetWindowSize(window, &windowWidth, &windowHeight); // get as base width and height
				glfwGetWindowContentScale(window, &scale, nullptr);
			}

			// scale window by DPI
			if (oldScale != scale) {
				windowWidth *= (scale / oldScale);
				windowHeight *= (scale / oldScale);
				glfwSetWindowSize(window, windowWidth, windowHeight); // rescale window by DPI
			}

			// on resizing (include DPI scaling)
			if (oldWidth != windowWidth || oldHeight != windowHeight) {
				glfwGetFramebufferSize(window, &canvasWidth, &canvasHeight);
				this->resize(canvasWidth, canvasHeight); // resize canvas
			}

			// do ray tracing
			this->process();

			// Measure speed
			double currentTime = glfwGetTime();
			if (currentTime - lastTime >= measureSeconds) {
				std::cout << "FPS: " << 1.f / (currentTime - prevFrameTime) << std::endl;
				lastTime += measureSeconds;
			}
			prevFrameTime = currentTime;

			// swapchain
			glfwSwapBuffers(window);
		}
	}

	void PathTracerApplication::saveHdr(std::string name) {
		psm::Pipeline::HdrImage image = rays->snapHdr();

		// allocate RGBAF
		FIBITMAP * btm = FreeImage_AllocateT(FIT_RGBAF, image.width, image.height);

		// copy HDR data
		int ti = 0;
		for (int r = 0; r < image.height; r++) {
			auto row = FreeImage_GetScanLine(btm, r);
			memcpy(row, image.image + r * 4 * image.width, image.width * sizeof(float) * 4);
		}

		// convert as 48 bits
		//btm = FreeImage_ConvertToRGBF(btm);

		// save HDR
		FreeImage_Save(FIF_EXR, btm, name.c_str(), EXR_FLOAT | EXR_PIZ);
		FreeImage_Unload(btm);
	}

    // key downs
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
        if (key == GLFW_KEY_L) keys[kL] = true;
    }

    // key release
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
        if (key == GLFW_KEY_L) {
            saveHdr("snapshots/hdr_snapshot_" + std::to_string(img_counter++) + ".exr");
            keys[kL] = false;
        }
    }

    // mouse moving and pressing
    void PathTracerApplication::mousePress(const int32_t& button) { if (button == GLFW_MOUSE_BUTTON_LEFT) lbutton = true; }
    void PathTracerApplication::mouseRelease(const int32_t& button) { if (button == GLFW_MOUSE_BUTTON_LEFT) lbutton = false; }
    void PathTracerApplication::mouseMove(const double& x, const double& y) { mousepos.x = x, mousepos.y = y; }

    // resize buffers and canvas functions
    void PathTracerApplication::resizeBuffers(const int32_t& width, const int32_t& height) { rays->resizeBuffers(width, height); }
    void PathTracerApplication::resize(const int32_t& width, const int32_t& height) { rays->resize(width, height); }
};

