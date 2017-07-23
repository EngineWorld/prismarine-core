RWStructuredBuffer<Leaf> Leafs : register(u0);
RWStructuredBuffer<uint> Mortoncodes : register(u1);
RWStructuredBuffer<int> MortoncodesIndices : register(u2);
RWStructuredBuffer<int> Range : register(u3);
RWStructuredBuffer<HlbvhNode> Nodes : register(u4);
RWStructuredBuffer<int> Flags : register(u5);
