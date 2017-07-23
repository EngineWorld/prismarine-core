
static const int At = 0;
static const int Rt = 1;
static const int Qt = 2;
static const int Ut = 3;
static const int Ct = 4;

RWStructuredBuffer<Ray> rayBuf : register(u0);
RWStructuredBuffer<Hit> hitBuf : register(u1);
RWStructuredBuffer<Texel> texelBuf : register(u2);
RWStructuredBuffer<int> activedBuf : register(u6);
RWStructuredBuffer<int> collBuf : register(u7);
RWStructuredBuffer<int> freedBuf : register(u8);
RWStructuredBuffer<int> availBuf : register(u14);
RWStructuredBuffer<int> arcounter : register(u20);
RWStructuredBuffer<ColorChain> chBuf : register(u21);


void _collect(inout Ray ray) {
    float4 color = max(ray.final, float4(0.f, 0.f, 0.f, 0.f));
    float amplitude = mlength(color.xyz);
    if (amplitude >= 0.00001f) {
        int idx = 0;
        InterlockedAdd(arcounter[Ct], 1, idx);
        int prev = 0;
        InterlockedExchange(texelBuf[ray.texel].EXT.y, idx, prev);
        ColorChain ch = chBuf[idx];
        ch.color = ray.final;
        ch.cdata.x = prev;
        chBuf[idx] = ch;
    }
    ray.final.xyzw = float4(0.f, 0.f, 0.f, 0.f);
}

void storeHit(in int hitIndex, inout Hit hit) {
    if (hitIndex == -1 || hitIndex == LONGEST || hitIndex >= rayBlock[0].currentRayLimit) {
        return;
    }
    hitBuf[hitIndex] = hit;
}

void storeHit(inout Ray ray, inout Hit hit) {
    storeHit(ray.idx, hit);
}


int addRayToList(in Ray ray){
    int rayIndex = ray.idx;
    int actived = -1;
    if (ray.actived == 1) {
        int act = 0;
        InterlockedAdd(arcounter[At], 1, act);
        collBuf[act] = rayIndex; actived = act;
    } else { // if not actived, why need?
        int freed = 0;
        InterlockedAdd(arcounter[Qt], 1, freed);
        freedBuf[freed] = rayIndex;
    }
    return actived;
}

int addRayToList(in Ray ray, in int act){
    int rayIndex = ray.idx;
    int actived = -1;
    if (ray.actived == 1) {
        collBuf[act] = rayIndex; actived = act;
    }
    return actived;
}


void storeRay(in int rayIndex, inout Ray ray) {
    if (rayIndex == -1 || rayIndex == LONGEST || rayIndex >= rayBlock[0].currentRayLimit) {
        ray.actived = 0;
        return;
    }
    _collect(ray);

    ray.idx = rayIndex;
    rayBuf[rayIndex] = ray;
}

void storeRay(inout Ray ray) {
    storeRay(ray.idx, ray);
}

int createRayStrict(inout Ray original, in int idx, in int rayIndex) {
    if (rayIndex == -1 || rayIndex == LONGEST || rayIndex >= rayBlock[0].currentRayLimit) {
        return rayIndex;
    }

    bool invalidRay = 
        original.actived < 1 || 
        original.bounce <= 0 || 
        mlength(original.color.xyz) < 0.0001f;

    if (invalidRay) {
        return rayIndex; 
    }

    Ray ray = original;
    ray.bounce -= 1;
    ray.idx = rayIndex;
    ray.texel = idx;

    // mark as unusual
    if (invalidRay) {
        ray.actived = 0;
    }

    Hit hit;
    if (original.idx != LONGEST) {
        hit = hitBuf[original.idx];
    } else {
        hit.normal = float4(0.f, 0.f, 0.f, 0.f);
        hit.tangent = float4(0.f, 0.f, 0.f, 0.f);
        hit.vmods = float4(0.f, 0.f, 0.f, 0.f);
        hit.triangleID = LONGEST;
        hit.materialID = LONGEST;
    }
    hit.shaded = 1;

    hitBuf[rayIndex] = hit;
    rayBuf[rayIndex] = ray;

    addRayToList(ray);
    return rayIndex;
}

int createRayStrict(inout Ray original, in int rayIndex) {
    return createRayStrict(original, original.texel, rayIndex);
}

int createRay(inout Ray original, in int idx) {
    _collect(original);

    bool invalidRay = 
        original.actived < 1 || 
        original.bounce <= 0 || 
        mlength(original.color.xyz) < 0.0001f;

    if (invalidRay) {
        return -1; 
    }

    int rayIndex = -1;
    int iterations = 1;
    int freed = 0;
    
    while (freed >= 0 && iterations >= 0) {
        iterations--;

        InterlockedMax(arcounter[Ut], 0);
        int freed = 0; InterlockedAdd(arcounter[Ut], -1, freed); freed -= 1;
        InterlockedMax(arcounter[Ut], 0);

        if (
            freed >= 0 && 
            availBuf[freed] != 0xFFFFFFFF && 
            availBuf[freed] != 0 && 
            availBuf[freed] != -1
        ) {
            rayIndex = availBuf[freed];
            break;
        }
    }

    if (rayIndex == -1) {
        InterlockedAdd(arcounter[Rt], 1, rayIndex); 
    }

    return createRayStrict(original, idx, rayIndex);
}

int createRayIdx(inout Ray original, in int idx, in int rayIndex) {
    _collect(original);

    bool invalidRay = 
        original.actived < 1 || 
        original.bounce <= 0 || 
        mlength(original.color.xyz) < 0.0001f;

    if (invalidRay) {
        return -1; 
    }
    
    InterlockedMax(arcounter[Rt], rayIndex+1);
    return createRayStrict(original, idx, rayIndex);
}

int createRay(in Ray original) {
    return createRay(original, original.texel);
}

int createRay(in int idx) {
    Ray newRay;
    return createRay(newRay, idx);
}

Ray fetchRayDirect(in int texel) {
    return rayBuf[texel];
}

Hit fetchHitDirect(in int texel) {
    return hitBuf[texel];
}

Hit fetchHit(in Ray ray){
    return hitBuf[ray.idx];
}
