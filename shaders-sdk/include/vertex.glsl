#ifndef _VERTEX_H
#define _VERTEX_H

layout ( std430, binding = 3 ) volatile buffer GeomVertexSSBO {VboDataStride verts[];};
//layout ( std430, binding = 4 ) volatile buffer GeomIndicesSSBO {int indics[];};
layout ( std430, binding = 5 ) volatile buffer GeomMaterialsSSBO {int mats[];};

#endif
