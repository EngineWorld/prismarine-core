
// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md


// Morton codes and geometry counters
layout ( std430, binding = 0 ) volatile buffer MortoncodesBlock {
    MORTONTYPE Mortoncodes[];
};

layout ( std430, binding = 1 ) volatile buffer IndicesBlock {
    int MortoncodesIndices[];
};

layout ( std430, binding = 2 ) readonly buffer NumBlock {
    int Range[1];
};

layout ( std430, binding = 3 ) volatile buffer LeafBlock {
    HlbvhNode Leafs[];
};

// BVH nodes
layout ( std430, binding = 4 ) volatile buffer NodesBlock {
    HlbvhNode Nodes[];
};

layout ( std430, binding = 5 ) volatile buffer FlagsBlock {
    int Flags[];
};

layout ( std430, binding = 6 ) volatile buffer ActivesBlock {
    int Actives[];
};

layout ( std430, binding = 7 ) volatile buffer ChildBuffer {
    int LeafIndices[];
};
