#pragma once

#include "includes.hpp"
#include "utils.hpp"
#include "tracer.hpp"

namespace Paper {
    class Controller : public PTObject {
        bool monteCarlo = true;

    public:
        glm::dvec3 eye = glm::dvec3(0.0f, 6.0f, 6.0f);
        glm::dvec3 view = glm::dvec3(0.0f, 2.0f, 0.0f);
        glm::dvec2 mposition;
        Tracer * raysp;

        glm::dmat4 project() {
#ifdef USE_CAD_SYSTEM
            return glm::lookAt(eye, view, glm::dvec3(0.0f, 0.0f, 1.0f));
#elif USE_180_SYSTEM
            return glm::lookAt(eye, view, glm::dvec3(0.0f, -1.0f, 0.0f));
#else
            return glm::lookAt(eye, view, glm::dvec3(0.0f, 1.0f, 0.0f));
#endif
        }

        void setRays(Tracer * r) {
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

            eye  = (unviewm * glm::vec4(ca, 1.0f)).xyz();
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
}
