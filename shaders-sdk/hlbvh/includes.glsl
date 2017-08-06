// Morton codes and geometry counters
layout ( std430, binding = 0 ) restrict buffer MortoncodesBlock {
    uint Mortoncodes[];
};

layout ( std430, binding = 1 ) restrict buffer IndicesBlock {
    int MortoncodesIndices[];
};

layout ( std430, binding = 2 ) readonly buffer NumBlock {
    int Range[1];
};

layout ( std430, binding = 3 ) restrict buffer LeafBlock {
    HlbvhNode Leafs[];
};

// BVH nodes
layout ( std430, binding = 4 ) restrict buffer NodesBlock {
    HlbvhNode Nodes[];
};

layout ( std430, binding = 5 ) restrict buffer FlagsBlock {
    int Flags[];
};

layout ( std430, binding = 6 ) restrict buffer ActivesBlock {
    int Actives[];
};
