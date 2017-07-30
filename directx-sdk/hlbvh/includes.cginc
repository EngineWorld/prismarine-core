
RWStructuredBuffer<uint> Mortoncodes : register(u0);
RWStructuredBuffer<int> MortoncodesIndices : register(u1);
RWStructuredBuffer<int> Range : register(u2);
RWStructuredBuffer<Leaf> Leafs : register(u3);
RWStructuredBuffer<HlbvhNode> Nodes : register(u4);
RWStructuredBuffer<int> Flags : register(u5);
