
layout ( std430, binding = 0 ) buffer LeafBlock {
    Leaf Leafs[];
};

layout ( std430, binding = 1 ) buffer MortoncodesBlock {
    uint Mortoncodes[];
};

layout ( std430, binding = 2 ) buffer IndicesBlock {
    int MortoncodesIndices[];
};

layout ( std430, binding = 3 ) readonly buffer NumBlock {
    int Range[1];
};

layout ( std430, binding = 4 ) buffer NodesBlock {
    HlbvhNode Nodes[];
};

layout ( std430, binding = 5 ) buffer FlagsBlock {
    int Flags[];
};
