#include "../include/constants.hlsl"
#include "../include/structs.hlsl"
#include "../include/uniforms.hlsl"
#include "./includes.hlsl"

#define LEAFIDX(i) (Range[0] + i)
#define NODEIDX(i) (clamp(i, 1, Range[0]-1))
#define LOCAL_SIZE 1024

int nlz (in uint x) {
    return 31-firstbithigh(x);
}

int nlz (in int x) {
    return 31-firstbithigh(uint(x));
}

int findSplit( int first, int last)
{
    uint firstCode = Mortoncodes[first];
    uint lastCode = Mortoncodes[last];
    if (firstCode == lastCode) {
        return (first + last) >> 1;
    }

    int commonPrefix = nlz(firstCode ^ lastCode);
    int split = first;
    int step = last - first;
    for (int i=0;i<8192;i++) {
        step = (step + 1) >> 1;
        int newSplit = split + step;
        if (newSplit < last) {
             uint splitCode = Mortoncodes[newSplit];
             int splitPrefix = nlz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
        if (step <= 1) break; 
    }

    return clamp(split, 0, Range[0]-2);
}

groupshared int lRange[2];
groupshared int lCounter;

[numthreads(LOCAL_SIZE, 1, 1)]
void CSMain( uint3 WorkGroupID : SV_GroupID, uint3 LocalInvocationID  : SV_GroupThreadID, uint3 GlobalInvocationID : SV_DispatchThreadID)
{
    if (WorkGroupID.x > 0) return; // not supported
    uint threadID = uint(LocalInvocationID.x);

    if (threadID == 0) {
        lCounter = 1;
        lRange[0] = 0;
        lRange[1] = 1;

        HlbvhNode node = Nodes[threadID];
        node.box.mn = (0.0f).xxxx;
        node.box.mx = (0.0f).xxxx;
        node.pdata.xy = int2(0, Range[0]-1);
        node.pdata.zw = int2(-1, -1);
        
        Nodes[threadID] = node;
        Flags[threadID] = 0;
    }
    
    AllMemoryBarrierWithGroupSync();

    for (int h=0;h<256;h++) {
        uint workSize = min((lRange[1] <= lRange[0]) ? 0 : (((lRange[1] - lRange[0]) - 1) / LOCAL_SIZE + 1), 8192);
        for (int i=0;i<workSize;i++) {
            
            uint sWorkID = LOCAL_SIZE * i;
            uint prID = lRange[0] + (sWorkID + threadID);

            if (prID < lRange[1]) {
                HlbvhNode parentNode = Nodes[prID];

                if (parentNode.pdata.x != parentNode.pdata.y) {
                    // find split
                    int split = findSplit(parentNode.pdata.x, parentNode.pdata.y);

                    // add to list
                    int hid = 0;
                    InterlockedAdd(lCounter, 2, hid);
                    int lid = hid+0, rid = hid+1;

                    HlbvhNode leftNode = Nodes[lid];
                    leftNode.box.mn = (0.0f).xxxx;
                    leftNode.box.mx = (0.0f).xxxx;
                    leftNode.pdata.xy = int2(parentNode.pdata.x, split+0);
                    leftNode.pdata.zw = int2(prID, -1);
                    
                    HlbvhNode rightNode = Nodes[rid];
                    rightNode.box.mn = (0.0f).xxxx;
                    rightNode.box.mx = (0.0f).xxxx;
                    rightNode.pdata.xy = int2(split+1, parentNode.pdata.y);
                    rightNode.pdata.zw = int2(prID, -1);
                    
                    // left node
                    Nodes[lid] = leftNode;
                    Flags[lid] = 0;

                    // right node
                    Nodes[rid] = rightNode;
                    Flags[rid] = 0;

                    // connect with childrens
                    parentNode.pdata.xy = (lid, rid);
                    Nodes[prID] = parentNode;
                } else {
                    uint leafID = MortoncodesIndices[parentNode.pdata.x];
                    
                    // make leafs
                    Leaf ourLeaf = Leafs[leafID];
                    ourLeaf.pdata.z = int(prID);
                    Leafs[leafID] = ourLeaf;

                    // load leaf data
                    int parentTmp = parentNode.pdata.z;
                    parentNode.box.mn = ourLeaf.box.mn;
                    parentNode.box.mx = ourLeaf.box.mx;
                    parentNode.pdata = ourLeaf.pdata;
                    parentNode.pdata.z = parentTmp;
                    Nodes[prID] = parentNode;
                }
            }
        }

        AllMemoryBarrierWithGroupSync();
        if (threadID == 0) {
            lRange[0] = lRange[1];
            lRange[1] = lCounter;
        }
        AllMemoryBarrierWithGroupSync();

        if (lRange[1] <= lRange[0]) {
            break;
        }
    }
}
