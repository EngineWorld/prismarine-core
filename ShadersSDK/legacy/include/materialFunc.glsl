#ifndef _MATERIALFUNC_H
#define _MATERIALFUNC_H

//#include "./pivot.glsl"
//#include "./ggx.glsl"

#define GAP PZERO*1.0f


// other parameters TODO
struct SurfaceData {
    vec3 normal;
    float height;

    vec4 emission;
    vec4 albedo;
    vec4 specular;
    
    float metallic;
    float roughness;

    int culling;
};

struct Material {
    vec4 diffuse;
    vec4 specular;
    vec4 transmission;
    vec4 emissive;

    float ior;
    float roughness;
    float alpharef;
    float unk0f;

    uint diffusePart;
    uint specularPart;
    uint bumpPart;
    uint emissivePart;

    int flags;
    int alphafunc;
    int binding;
    int unk0i;

    ivec4 iModifiers0;
};


const uint MAX_TEXTURES = 72;
layout ( binding = 15, std430 ) readonly buffer MaterialsSSBO {Material submats[];};


//layout ( binding = 5 ) uniform samplerCube skybox[1];
layout ( binding = 5 ) uniform sampler2D skybox[1];

#ifdef USE_OPENGL
#ifdef USE_BINDLESS
//layout ( location = 1, bindless_sampler ) uniform sampler2D samplers[MAX_TEXTURES];
layout ( binding = 16 ) readonly buffer Textures { sampler2D samplers[]; };
#else
layout ( location = 1 ) uniform sampler2D samplers[MAX_TEXTURES];
#endif
#else
layout ( binding = 6, set = 1 ) uniform sampler2D samplers[MAX_TEXTURES]; // vulkan API type (future)
#endif



vec4 cubic(in float v){
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}

vec4 textureBicubic(in sampler2D sampler, in vec2 texCoords){
    vec2 texSize = textureSize(sampler, 0);
    vec2 invTexSize = 1.0 / texSize;
    texCoords = texCoords * texSize - 0.5;

    vec2 fxy = fract(texCoords);
    texCoords -= fxy;

    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    vec4 c = texCoords.xxyy + vec2 (-0.5, +1.5).xyxy;
    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;

    offset *= invTexSize.xxyy;

    vec4 sample0 = texture(sampler, offset.xz);
    vec4 sample1 = texture(sampler, offset.yz);
    vec4 sample2 = texture(sampler, offset.xw);
    vec4 sample3 = texture(sampler, offset.yw);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

vec4 readEnv(in vec3 r) {
    vec3 nr = normalize(r);
    return texture(skybox[0], vec2(fma(vec2(atan(nr.z, nr.x), asin(nr.y) * 2.0f) / PI, vec2(0.5), vec2(0.5))));
}

bool haveProp(in Material material, in int prop) {
    return (material.flags & prop) > 0;
}

bool haveProp(in int flags, in int prop) {
    return (flags & prop) > 0;
}

bool validateTexture(in uint binding){
    return binding != 0 && binding != LONGEST && binding < MAX_TEXTURES && textureSize(samplers[binding], 0).x > 0;
}

vec4 fetchTexture(in uint binding, in vec2 texcoord){
    return texture(samplers[binding], texcoord);
}

vec4 fetchTexture(in uint binding, in vec2 texcoord, in ivec2 offset){
    return texture(samplers[binding], texcoord + vec2(offset) / textureSize(samplers[binding], 0));
}

float computeFresnel(in vec3 normal, in vec3 indc, in float n1, in float n2){
    float cosi = dot(normal,  normalize(indc));
    float cost = dot(normal,  normalize(refract(indc, normal, n1 / n2)));
    float Rs = fma(n1, cosi, - n2 * cost) / fma(n1, cosi, n2 * cost);
    float Rp = fma(n1, cost, - n2 * cosi) / fma(n1, cost, n2 * cosi);
    return sqrt(clamp(sqlen(vec2(Rs, Rp)) * 0.5f, 0.0f, 1.0f));
}

vec3 glossy(in vec3 dir, in vec3 normal, in float refli) {
    return normalize(mix(normalize(dir), randomCosine(normal), clamp(sqrt(random()) * refli, 0.0f, 1.0f)));
}

Ray reflection(in Ray ray, in Hit hit, in vec3 color, in vec3 normal, in float refly){
    ray.direct.xyz = normalize(mix(reflect(ray.direct.xyz, normal), randomCosine(normal), clamp(refly * random(), 0.0f, 1.0f)));
    ray.color.xyz *= color;
    ray.params.x = (SUNLIGHT_CAUSTICS ? true : ray.params.z < 1) ? 0 : 1;
    ray.bounce = min(3, ray.bounce);
    ray.origin.xyz = fma(faceforward(hit.normal.xyz, ray.direct.xyz, -hit.normal.xyz), vec3(GAP), ray.origin.xyz); // padding
    ray.actived = ray.params.w == 1 ? 0 : ray.actived;
    return ray;
}

Ray refraction(in Ray ray, in Hit hit, in vec3 color, in vec3 normal, in float inior, in float outior, in float glossiness){
     vec3 refrDir = normalize(  refract(ray.direct.xyz, normal, inior / outior)  );
     bool refrc = equalF(inior, outior);

#ifdef REFRACTION_SKIP_SUN

    ray.bounce += 1;
    if (ray.params.w < 1) {
        ray.direct.xyz = refrDir;
    }

#else
    
    ray.direct.xyz = refrDir;
    if (!refrc) ray.params.x = (SUNLIGHT_CAUSTICS ? true : ray.params.z < 1) ? 0 : 1; // can be lighted by direct
    if (ray.params.w < 1 || refrc) { 
        ray.bounce += 1;
    }
    
#endif

    if (!refrc) ray.origin.xyz = fma(faceforward(hit.normal.xyz, ray.direct.xyz, -hit.normal.xyz), vec3(GAP), ray.origin.xyz); // padding
    ray.color.xyz *= color;
    return ray;
}

vec3 lightCenter(in int i){
    vec3 playerCenter = vec3(0.0f);
    vec3 lvec = normalize(lightUniform.lightNode[i].lightVector.xyz) * (lightUniform.lightNode[i].lightVector.y < 0.0f ? -1.0f : 1.0f);
    return fma(lvec, vec3(lightUniform.lightNode[i].lightVector.w), (lightUniform.lightNode[i].lightOffset.xyz + playerCenter.xyz));
}

vec3 sLight(in int i){
    return fma(randomDirectionInSphere(), vec3(lightUniform.lightNode[i].lightColor.w), lightCenter(i));
}

int applyLight(in Ray directRay, inout Ray ray, in vec3 normal){
#ifdef DIRECT_LIGHT
    directRay.actived = (ray.params.w == 1 || dot(normal, directRay.direct.xyz) < 0.f) ? 0 : directRay.actived;
    ray.params.x = 1;
    return createRay(directRay);
#else 
    return -1;
#endif
}

float intersectSphere(in vec3 origin, in vec3 ray, in vec3 sphereCenter, in float sphereRadius) {
    vec3 toSphere = origin - sphereCenter;
    float a = dot(ray, ray);
    float b = 2.0f * dot(toSphere, ray);
    float c = dot(toSphere, toSphere) - sphereRadius*sphereRadius;
    float discriminant = fma(b,b,-4.0f*a*c);
    float t = INFINITY;
    if(discriminant > 0.0f) {
        float da = 0.5f / a;
        float t1 = (-b - sqrt(discriminant)) * da;
        float t2 = (-b + sqrt(discriminant)) * da;
        float mn = min(t1, t2);
        float mx = max(t1, t2);
        t = mx >= 0.0f ? (mn >= 0.0f ? mn : mx) : t;
    }
    return t;
}

Ray directLight(in int i, in Ray directRay, in Hit hit, in vec3 color, in vec3 normal){
    directRay.bounce = min(1, directRay.bounce);
    directRay.actived = directRay.params.w == 1 ? 0 : directRay.actived;
    directRay.params.w = 1;
    directRay.params.x = 0;
    directRay.params.y = i;
    vec3 ltr = lightCenter(i).xyz-directRay.origin.xyz;
    vec3 ldirect = normalize(sLight(i) - directRay.origin.xyz);
    float cos_a_max = sqrt(1.f - clamp(lightUniform.lightNode[i].lightColor.w * lightUniform.lightNode[i].lightColor.w / sqlen(ltr), 0.0f, 1.0f));
    float diffuseWeight = clamp(dot(ldirect, normal), 0.0f, 1.0f);
    directRay.direct.xyz = ldirect;
    directRay.color.xyz *= color * diffuseWeight * ((1.0f - cos_a_max) * 2.0f);
    return directRay;
}

Ray diffuse(in Ray ray, in Hit hit, in vec3 color, in vec3 normal){
    ray.actived = ray.params.w == 1 ? 0 : ray.actived;
    ray.color.xyz *= color;
    ray.direct.xyz = normalize(randomCosine(normal));
    ray.bounce = min(2, ray.bounce);
    ray.params.z = 1; // hit on diffuse
    ray.params.x = 0; // enable sunlight
    ray.origin.xyz = fma(faceforward(hit.normal.xyz, ray.direct.xyz, -hit.normal.xyz), vec3(GAP), ray.origin.xyz); // padding
    return ray;
}

Ray promised(in Ray ray, in Hit hit, in vec3 normal){
    ray.bounce += 1;
    //ray.origin.xyz = fma(faceforward(hit.normal.xyz, ray.direct.xyz, -hit.normal.xyz), vec3(GAP), ray.origin.xyz); // padding
    // because transparency and baked will processing in fly, it may required
    return ray;
}

Ray emissive(in Ray ray, in Hit hit, in vec3 color, in vec3 normal){
    ray.final.xyz = max(ray.color.xyz * color, vec3(0.0f));
    ray.final = ray.params.w == 1 ? vec4(0.0f) : max(ray.final, vec4(0.0f));
    ray.color.xyz *= 0.0f;
    ray.direct.xyz = normalize(randomCosine(normal));
    ray.actived = 0;
    ray.params.x = 1;
    ray.origin.xyz = fma(faceforward(hit.normal.xyz, ray.direct.xyz, -hit.normal.xyz), vec3(GAP), ray.origin.xyz); // padding
    return ray;
}

int emitRay(in Ray directRay, in Hit hit, in vec3 normal, in float coef){
    directRay.color.xyz *= coef;
    directRay.final.xyz *= coef;
    return createRay(directRay);
}

bool doesCubeIntersectSphere(in vec3 C1, in vec3 C2, in vec3 S, in float R)
{
    float dist_squared = R * R;
    if (S.x < C1.x) dist_squared -= sqlen(S.x - C1.x);
    else if (S.x > C2.x) dist_squared -= sqlen(S.x - C2.x);
    if (S.y < C1.y) dist_squared -= sqlen(S.y - C1.y);
    else if (S.y > C2.y) dist_squared -= sqlen(S.y - C2.y);
    if (S.z < C1.z) dist_squared -= sqlen(S.z - C1.z);
    else if (S.z > C2.z) dist_squared -= sqlen(S.z - C2.z);
    return dist_squared > 0;
}

vec3 getLightColor(in int lc){
    return max(lightUniform.lightNode[lc].lightColor.xyz, vec3(0.f));
}

#endif
