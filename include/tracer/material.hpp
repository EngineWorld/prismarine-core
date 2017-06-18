#pragma once

#include "includes.hpp"
#include "utils.hpp"
#include <map>
#include <algorithm>

namespace Paper {
    class Material : public PTObject {
        friend class Tracer;

    public:

        struct Submat {
            glm::vec4 diffuse = glm::vec4(0.0f);
            glm::vec4 specular = glm::vec4(0.0f);
            glm::vec4 transmission = glm::vec4(0.0f);
            glm::vec4 emissive = glm::vec4(0.0f);

            float ior = 1.0f;
            float reflectivity = 0.0001f;
            float alpharef = 0.0f;
            float unk0f = 0.0f;

            uint32_t diffusePart = 0;
            uint32_t specularPart = 0;
            uint32_t bumpPart = 0;
            uint32_t emissivePart = 0;

            int32_t flags = 0;
            int32_t alphafunc = 0;
            int32_t binding = 0;
            int32_t unk0i = 0;

            glm::ivec4 iModifiers0 = glm::ivec4(0);
        };

    private:
        int32_t materialID = 0;
        GLuint mats = -1;

        void init();

    public:
        std::vector<Submat> submats;
        std::vector<uint32_t> samplers;
        std::vector<uint32_t> freedomSamplers;
        std::map<std::string, uint32_t> texnames;

        Material() {
			submats = std::vector<Submat>(0); // init
            samplers = std::vector<uint32_t>(0);
            samplers.push_back(0);

            freedomSamplers = std::vector<uint32_t>(0);
            texnames = std::map<std::string, uint32_t>();
            init();
        }


        void clearSubmats() {
            submats.resize(0);
        }

        size_t addSubmat(const Submat * submat) {
            size_t idx = submats.size();
            submats.push_back(*submat);
            return idx;
        }

        size_t addSubmat(const Submat &submat) {
            return this->addSubmat(&submat);
        }

        void setSumbat(const size_t& i, const Submat &submat) {
            if (submats.size() <= i) submats.resize(i+1);
            submats[i] = submat;
        }

        void loadToVGA();
        void bindWithContext(GLuint & prog);

		void freeTextureByGL(const GLuint& idx);
        void freeTexture(const uint32_t& idx);

        uint32_t loadTexture(std::string tex, bool force_write = false);
        uint32_t loadTexture(const GLuint & gltexture);

        GLuint getGLTexture(const uint32_t & idx);
        uint32_t getTexture(const GLuint & idx);
    };
}

#include "material.inl"