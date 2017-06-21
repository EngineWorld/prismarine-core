#ifndef _MATERIALFUNC_H
#define _MATERIALFUNC_H

#define GAP PZERO*1.01f

struct Submat {
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

layout ( location = 0 ) uniform sampler2D samplers[128];
layout ( binding=15, std430 ) readonly buffer MaterialsSSBO {Submat submats[];};

bool haveProp(in Submat material, in int prop) {
    return (material.flags & prop) > 0;
}

bool haveProp(in int flags, in int prop) {
    return (flags & prop) > 0;
}

bool validateTexture(in uint binding){
    //return smp != 0;//0xFFFFFFFF;
    //return textureSize(samplers[binding], 0).x > 0;
    return binding != 0;
}

vec4 fetchPart(in uint binding, in vec2 texcoord){
    //return texelFetchWrap(binding, texcoord);
    //return texelFetchLinear(binding, texcoord);
#ifdef FREEIMAGE_STYLE
    return texture(samplers[binding], texcoord).bgra;
#else
    return texture(samplers[binding], texcoord);
#endif
}

vec4 fetchPart(in uint binding, in vec2 texcoord, in ivec2 offset){
    //return texelFetchWrap(binding, texcoord, offset);
    //return texelFetchLinear(binding, texcoord, offset);
#ifdef FREEIMAGE_STYLE
    return texture(samplers[binding], texcoord + vec2(offset) / textureSize(samplers[binding], 0)).bgra;
#else
    return texture(samplers[binding], texcoord + vec2(offset) / textureSize(samplers[binding], 0));
#endif
}

vec4 fetchSpecular(in Submat mat, in vec2 texcoord, in vec3 direct, in vec3 normal){
    vec4 specular = mat.specular;
    if (validateTexture(mat.specularPart)) {
        specular = fetchPart(mat.specularPart, texcoord);
    }
    specular.xyz = pow(specular.xyz, vec3(GAMMA));
    //specular = pow(specular, vec4(GAMMA));
    return specular;
}

vec4 fetchEmissive(in Submat mat, in vec2 texcoord, in vec3 direct, in vec3 normal){
    vec4 emission = vec4(0.0f);
    if(!validateTexture(mat.emissivePart)) {
        emission = mat.emissive;
    } else {
        emission = fetchPart(mat.emissivePart, texcoord);
    }
    return emission;
}

vec4 fetchTransmission(in Submat mat, in vec2 texcoord, in vec3 direct, in vec3 normal){
    return mat.transmission;
}

vec4 fetchNormal(in Submat mat, in vec2 texcoord, in vec3 direct, in vec3 normal){
    if(!validateTexture(mat.bumpPart)) {
        return vec4(0.5f, 0.5f, 1.0f, 1.0f);
    } else {
        return fetchPart(mat.bumpPart, vec2(texcoord.x, texcoord.y));
    }
    return vec4(0.0f);
}

vec4 fetchNormal(in Submat mat, in vec2 texcoord, in ivec2 offset, in vec3 direct, in vec3 normal){
    if(!validateTexture(mat.bumpPart)) {
        return vec4(0.5f, 0.5f, 1.0f, 1.0f);
    } else {
        return fetchPart(mat.bumpPart, vec2(texcoord.x, texcoord.y), offset);
    }
    return vec4(0.0f);
}

vec3 getNormalMapping(in Submat mat, vec2 texcoordi, in vec3 direct, in vec3 normal) {
    vec3 tc = fetchNormal(mat, texcoordi, direct, normal).xyz;
    if(equalF(tc.x, tc.y) && equalF(tc.x, tc.z)){
        const ivec3 off = ivec3(0,0,1);
        const float size = 1.0f;
        const float pike = 2.0f;
        vec3 p00 = vec3(0.0f, 0.0f, fetchNormal(mat, texcoordi, off.yy, direct, normal).x * pike);
        vec3 p01 = vec3(size, 0.0f, fetchNormal(mat, texcoordi, off.zy, direct, normal).x * pike);
        vec3 p10 = vec3(0.0f, size, fetchNormal(mat, texcoordi, off.yz, direct, normal).x * pike);
        return normalize(cross(p10 - p00, p01 - p00));
    } else {
        return normalize(fma(tc, vec3(2.0f), vec3(-1.0f)));
    }
    return vec3(0.0, 0.0, 1.0);
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

void mixed(inout vec3 src, inout vec3 dst, in float coef){
    dst *= coef;
    src *= 1.0f - coef;
}

void mixed(inout vec3 src, inout vec3 dst, in vec3 coef){
    dst *= coef;
    src *= 1.0f - coef;
}

Ray reflection(in Ray newRay, in Hit hit, in vec3 color, in vec3 normal, in float refly){
    if (newRay.params.w == 1) return newRay;
    newRay.direct.xyz = mix(randomCosine(normal), reflect(newRay.direct.xyz, normal), refly);
    newRay.color.xyz *= color;
    newRay.params.x = SUNLIGHT_CAUSTICS ? 0 : 1;
    newRay.bounce = min(2, newRay.bounce); // easier mode
    newRay.origin.xyz = fma(faceforward(hit.normal.xyz, newRay.direct.xyz, -hit.normal.xyz), vec3(GAP), newRay.origin.xyz); // padding
    return newRay;
}

Ray refraction(in Ray newRay, in Hit hit, in vec3 color, in vec3 normal, in float inior, in float outior, in float glossiness){
    const vec3 refrDir = normalize(  refract(newRay.direct.xyz, normal, inior / outior)  );
    const bool refrc = equalF(inior, outior);

#ifdef REFRACTION_SKIP_SUN

    newRay.bounce += 1;
    if (newRay.params.w < 1) {
        newRay.direct.xyz = refrDir;
    }

#else
    
    newRay.direct.xyz = refrDir;
    if (!refrc) newRay.params.x = SUNLIGHT_CAUSTICS ? 0 : 1; // can be lighted by direct
    if (newRay.params.w < 1 || refrc) { 
        newRay.bounce += 1;
    }
    
#endif

    if (!refrc) newRay.origin.xyz = fma(faceforward(hit.normal.xyz, newRay.direct.xyz, -hit.normal.xyz), vec3(GAP), newRay.origin.xyz); // padding
    newRay.color.xyz *= color;
    return newRay;
}

Ray transformRay(in Ray directRay){
    directRay.origin.xyz = (inverse(RAY_BLOCK materialUniform.transformModifier) * vec4(directRay.origin.xyz, 1.0f)).xyz;
    directRay.direct.xyz = normalize((inverse(RAY_BLOCK materialUniform.transformModifier) * vec4(directRay.direct.xyz, 0.0f)).xyz);
    return directRay;
}


vec3 transformNormal(in vec3 norm){
    norm.xyz = normalize((vec4(norm, 0.0f) * (RAY_BLOCK materialUniform.transformModifier)).xyz);
    return norm;
}

vec3 unTransformNormal(in vec3 norm){
    norm.xyz = normalize((vec4(norm, 0.0f) * inverse(RAY_BLOCK materialUniform.transformModifier)).xyz);
    return norm;
}

Ray transformRay(in Ray directRay, inout float distModifier){
    directRay.origin.xyz = (inverse(RAY_BLOCK materialUniform.transformModifier) * vec4(directRay.origin.xyz, 1.0f)).xyz;

    const vec3 backup = directRay.direct.xyz;
    directRay.direct.xyz = (inverse(RAY_BLOCK materialUniform.transformModifier) * vec4(directRay.direct.xyz, 0.0f)).xyz;
    distModifier = length(directRay.direct.xyz) / length(backup);
    directRay.direct.xyz = normalize(directRay.direct.xyz);

    return directRay;
}

Ray unTransformRay(in Ray directRay){
    directRay.origin.xyz = (RAY_BLOCK materialUniform.transformModifier * vec4(directRay.origin.xyz, 1.0f)).xyz;
    directRay.direct.xyz = normalize((RAY_BLOCK materialUniform.transformModifier * vec4(directRay.direct.xyz, 0.0f)).xyz);
    return directRay;
}

vec3 lightCenter(in int i){
    const vec3 playerCenter = (inverse(RAY_BLOCK materialUniform.transformModifier) * vec4(vec3(0.0f), 1.0f)).xyz;
    const vec3 lvec = normalize(lightUniform[i].lightVector.xyz) * (lightUniform[i].lightVector.y < 0.0f ? -1.0f : 1.0f);
    return fma(lvec, vec3(lightUniform[i].lightVector.w), (lightUniform[i].lightOffset.xyz + playerCenter.xyz));
}

vec3 sLight(in int i){
    return fma(randomDirectionInSphere(), vec3(lightUniform[i].lightColor.w), lightCenter(i));
}

int applyLight(in Ray directRay, inout Ray newRay, in vec3 normal){
#ifdef DIRECT_LIGHT
    if (newRay.params.w == 1) return -1; // don't accept from shadow originals
    if (dot(normal, directRay.direct.xyz) < 0.f) return -1; // don't accept regret shadows
    newRay.params.x = 1;
    return createRay(directRay);
#else 
    return -1;
#endif
}

Ray directLight(in int i, in Ray directRay, in Hit hit, in vec3 color, in vec3 normal){
    if (directRay.params.w == 1) return directRay;
    directRay = transformRay(directRay);
    directRay.bounce = min(1, directRay.bounce);
    directRay.actived = 1;
    directRay.params.w = 1;
    directRay.params.x = 0;
    directRay.params.y = i;

    const float cos_a_max = sqrt(1.f - clamp(lightUniform[i].lightColor.w * lightUniform[i].lightColor.w / sqlen(lightCenter(i).xyz-directRay.origin.xyz), 0.f, 1.f));
    directRay.direct.xyz = normalize(sLight(i) - directRay.origin.xyz);
    directRay.color.xyz *= color * clamp(dot(directRay.direct.xyz, transformNormal(normal)), 0.0f, 1.0f) * ((1.0f - cos_a_max) * 2.0f);
    directRay = unTransformRay(directRay);
    return directRay;
}

Ray diffuse(in Ray newRay, in Hit hit, in vec3 color, in vec3 normal){
    if (newRay.params.w == 1) return newRay;
    newRay.color.xyz *= color;
    newRay.direct.xyz = randomCosine(normal);
    newRay.bounce = min(2, newRay.bounce);
    newRay.params.z = 1;
    newRay.params.x = 0;
    newRay.origin.xyz = fma(faceforward(hit.normal.xyz, newRay.direct.xyz, -hit.normal.xyz), vec3(GAP), newRay.origin.xyz); // padding
    return newRay;
}

Ray promised(in Ray newRay, in Hit hit, in vec3 normal){
    newRay.bounce += 1;
    return newRay;
}

Ray emissive(in Ray newRay, in Hit hit, in vec3 color, in vec3 normal){
    if (newRay.params.w == 1) return newRay;
    newRay.final.xyz = max(newRay.color.xyz * color, vec3(0.0f));
    newRay.final = max(newRay.final, vec4(0.0f));
    newRay.actived = 0;
    newRay.params.x = 1;
    newRay.origin.xyz = fma(faceforward(hit.normal.xyz, newRay.direct.xyz, -hit.normal.xyz), vec3(GAP), newRay.origin.xyz); // padding
    return newRay;
}

int emitRay(in Ray directRay, in Hit hit, in vec3 normal, in float coef){
    directRay.color.xyz *= coef;
    directRay.final.xyz *= coef;
    const int ps = createRay(directRay);
    storeHit(ps, hit);
    return ps;
}

vec3 lightCenterSky(in int i) {
    const vec3 playerCenter = (inverse(RAY_BLOCK materialUniform.transformModifier) * vec4(vec3(0.0f), 1.0f)).xyz;
    const vec3 lvec = normalize(lightUniform[i].lightVector.xyz) * 1000.0f;
    return lightUniform[i].lightOffset.xyz + lvec + playerCenter.xyz;
}

float intersectSphere(in vec3 origin, in vec3 ray, in vec3 sphereCenter, in float sphereRadius) {
    const vec3 toSphere = origin - sphereCenter;
    const float a = dot(ray, ray);
    const float b = 2.0f * dot(toSphere, ray);
    const float c = dot(toSphere, toSphere) - sphereRadius*sphereRadius;
    const float discriminant = fma(b,b,-4.0f*a*c);
    if(discriminant > 0.0f) {
        const float da = 0.5f / a;
        const float t1 = (-b - sqrt(discriminant)) * da;
        const float t2 = (-b + sqrt(discriminant)) * da;
        const float mn = min(t1, t2);
        const float mx = max(t1, t2);
        if (mn >= 0.0f) return mn; else
        if (mx >= 0.0f) return mx;
    }
    return INFINITY;
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
    return lightUniform[lc].lightColor.xyz;
}

#endif
