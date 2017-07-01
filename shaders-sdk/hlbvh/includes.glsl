
layout ( std430, binding = 0 ) readonly buffer NumBlock {
    ivec2 Range;
};

layout ( std430, binding = 1 ) volatile buffer MortoncodesBlock {
    uint Mortoncodes[];
};

layout ( std430, binding = 2 ) volatile buffer IndicesBlock {
    int MortoncodesIndices[];
};

layout ( std430, binding = 3 ) volatile buffer LeafBlock {
    Leaf Leafs[];
};

layout ( std430, binding = 4 ) volatile buffer NodesBlock {
    HlbvhNode Nodes[];
};

layout ( std430, binding = 5 ) volatile buffer FlagsBlock {
    int Flags[];
};
