#pragma once

#include "includes.hpp"
#include "utils.hpp"
#include "tracer.hpp"

namespace Paper {
    class Transformer : public PTObject {
        protected:
        std::vector<glm::dmat4> stack;
        glm::dmat4 current;
        
        public:
            float voffsetAccum;
            int32_t flags = 0;
            int32_t exflags = 0;
            glm::vec4 colormod = glm::vec4(1.0f);

            Transformer(){
                stack = std::vector<glm::dmat4>();
                reset();
            }
        
            glm::dmat4 getCurrent(){
                return glm::dmat4(current);
            }
        
            void reset() {
                stack.resize(0);
                current = glm::dmat4(1.0);
                push();
            }

            void multiply(glm::dmat4 mat) {
                current = current * mat;
            }

            void multiply(glm::mat4 mat) {
                current = current * glm::dmat4(mat);
            }

            void multiply(const float * m) {
                multiply(glm::make_mat4(m));
            }

            void multiply(const double * m) {
                multiply(glm::make_mat4(m));
            }

            void rotate(double angle, glm::dvec3 rot){
                current = glm::rotate(current, angle, rot);
            }

            void rotate(double angle, double x, double y, double z) {
                rotate(angle * M_PI / 180.0, glm::dvec3(x, y, z));
            }
        
            void translate(glm::dvec3 offset){
                current = glm::translate(current, offset);
            }

            void translate(double x, double y, double z) {
                translate(glm::dvec3(x, y, z));
            }
        
            void scale(glm::dvec3 size){
                current = glm::scale(current, size);
            }

            void scale(double x, double y, double z) {
                scale(glm::dvec3(x, y, z));
            }
        
            void identity(){
                current = glm::dmat4(1.0);
            }
        
            void push(){
                stack.push_back(current);
            }
        
            void pop(){
                if (stack.size() <= 0) {
                    current = glm::dmat4(1.0);
                }
                else {
                    current = stack[stack.size() - 1];
                    stack.pop_back();
                }
            }
    };
}