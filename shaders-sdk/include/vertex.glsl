#ifndef _VERTEX_H
#define _VERTEX_H

layout ( std430, binding = 3 ) buffer GeomVertexSSBO {VboDataStride verts[];};
//layout ( std430, binding = 4 ) buffer GeomIndicesSSBO {int indics[];};
layout ( std430, binding = 5 ) buffer GeomMaterialsSSBO {int mats[];};

#endif
