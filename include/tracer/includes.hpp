#pragma once

#define RAY_TRACING_ENGINE

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <ctime>
#include <chrono>
#include <array>
#include <random>
#include <memory>
#include <sstream>

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/component_wise.hpp"
#include "glm/gtx/rotate_vector.hpp"
#include "glad/glad.h"

#ifdef USE_FREEIMAGE
#include "external/include/FreeImage.h"
#endif

