#ifndef _MATERIALFUNC_H
#define _MATERIALFUNC_H

//#include "./pivot.glsl"
//#include "./ggx.glsl"

#define GAP PZERO*2.0f

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

RayRework reflection(in RayRework ray, in vec3 color, in vec3 normal, in float refly){
    ray.direct.xyz = normalize(mix(reflect(ray.direct.xyz, normal), randomCosine(normal), clamp(refly * random(), 0.0f, 1.0f)));
    ray.color.xyz *= color;
    //ray.origin.xyz = fma(ray.direct.xyz, vec3(GAP), ray.origin.xyz);
    ray.origin.xyz = fma(faceforward(normal, ray.direct.xyz, -normal), vec3(GAP), ray.origin.xyz); // padding

    RayDL(ray, (SUNLIGHT_CAUSTICS ? true : RayType(ray) == 1) ? 0 : 1); RayType(ray, 0);
    RayBounce(ray, min(3, RayBounce(ray)));
    RayActived(ray, RayType(ray) == 2 ? 0 : RayActived(ray));
    return ray;
}

RayRework refraction(in RayRework ray, in vec3 color, in vec3 normal, in float inior, in float outior, in float glossiness){
     vec3 refrDir = normalize(  refract(ray.direct.xyz, normal, inior / outior)  );
     bool refrc = equalF(inior, outior);

#ifdef REFRACTION_SKIP_SUN

    RayBounce(ray, RayBounce(ray)+1);
    if (RayType(ray) != 2) {
        ray.direct.xyz = refrDir;
    }

#else
    
    ray.direct.xyz = refrDir;
    if (!refrc) RayDL(ray, (SUNLIGHT_CAUSTICS ? true : RayType(ray) == 1) ? 0 : 1); RayType(ray, 0); // can be lighted by direct
    if (RayType(ray) != 2 || refrc) { 
        RayBounce(ray, RayBounce(ray)+1);
    }
    
#endif

    if (!refrc) {
        //ray.origin.xyz = fma(ray.direct.xyz, vec3(GAP), ray.origin.xyz);
        ray.origin.xyz = fma(faceforward(normal, ray.direct.xyz, -normal), vec3(GAP), ray.origin.xyz); // padding
    }
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

int applyLight(in RayRework directRay, inout RayRework ray, in vec3 normal){
#ifdef DIRECT_LIGHT
    RayActived(ray, (RayType(ray) == 2 || dot(normal, directRay.direct.xyz) < 0.f) ? 0 : RayActived(ray)); RayDL(ray, 0);
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

RayRework directLight(in int i, in RayRework directRay, in vec3 color, in vec3 normal){
    RayActived(directRay, RayType(directRay) == 2 ? 0 : RayActived(directRay));
    RayDL(directRay, 1);
    RayType(directRay, 2);
    RayTargetLight(directRay, i);
    RayBounce(directRay, min(1, RayBounce(directRay)));
    
    vec3 ltr = lightCenter(i).xyz-directRay.origin.xyz;
    vec3 ldirect = normalize(sLight(i) - directRay.origin.xyz);
    float cos_a_max = sqrt(1.f - clamp(lightUniform.lightNode[i].lightColor.w * lightUniform.lightNode[i].lightColor.w / sqlen(ltr), 0.0f, 1.0f));
    float diffuseWeight = clamp(dot(ldirect, normal), 0.0f, 1.0f);
    directRay.direct.xyz = ldirect;
    directRay.color.xyz *= color * diffuseWeight * ((1.0f - cos_a_max) * 2.0f);
    return directRay;
}

RayRework diffuse(in RayRework ray, in vec3 color, in vec3 normal){
    ray.color.xyz *= color;
    ray.direct.xyz = normalize(randomCosine(normal));
    //ray.origin.xyz = ray.origin.xyz = fma(ray.direct.xyz, vec3(GAP), ray.origin.xyz);
    ray.origin.xyz = fma(faceforward(normal, ray.direct.xyz, -normal), vec3(GAP), ray.origin.xyz); // padding
    RayActived(ray, RayType(ray) == 2 ? 0 : RayActived(ray));
    RayBounce(ray, min(2, RayBounce(ray)));
    RayType(ray, 1);
    RayDL(ray, 1);
    return ray;
}

RayRework promised(in RayRework ray, in vec3 normal){
    RayBounce(ray, RayBounce(ray)+1);
    //ray.origin.xyz = ray.origin.xyz = fma(ray.direct.xyz, vec3(GAP), ray.origin.xyz);
    ray.origin.xyz = fma(faceforward(normal, ray.direct.xyz, -normal), vec3(GAP), ray.origin.xyz); // padding
    return ray;
}

RayRework emissive(in RayRework ray, in vec3 color, in vec3 normal){
    ray.final.xyz = max(ray.color.xyz * color, vec3(0.0f));
    ray.final = RayType(ray) == 1 ? vec4(0.0f) : max(ray.final, vec4(0.0f));
    ray.color.xyz *= 0.0f;
    ray.direct.xyz = normalize(randomCosine(normal));
    //ray.origin.xyz = fma(ray.direct.xyz, vec3(GAP), ray.origin.xyz);
    ray.origin.xyz = fma(faceforward(normal, ray.direct.xyz, -normal), vec3(GAP), ray.origin.xyz); // padding

    RayActived(ray, 0);
    RayDL(ray, 0);
    
    return ray;
}

int emitRay(in RayRework directRay, in vec3 normal, in float coef){
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
