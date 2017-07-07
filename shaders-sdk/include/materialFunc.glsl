#ifndef _MATERIALFUNC_H
#define _MATERIALFUNC_H

#include "./pivot.glsl"
#include "./ggx.glsl"

#define GAP PZERO*1.0f

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

const uint MAX_TEXTURES = 31;
layout ( location = 0 ) uniform sampler2D samplers[MAX_TEXTURES];
layout ( binding = 31 ) uniform sampler2D u_PivotSampler;

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
    return binding != 0 && binding != LONGEST && binding >= 0 && binding < MAX_TEXTURES && textureSize(samplers[binding], 0).x > 0;
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
    return specular;
}

vec4 fetchEmissive(in Submat mat, in vec2 texcoord, in vec3 direct, in vec3 normal){
    vec4 emission = vec4(0.0f);
    if (validateTexture(mat.emissivePart)) {
        emission = fetchPart(mat.emissivePart, texcoord);
    }
    return emission;
}

vec4 fetchTransmission(in Submat mat, in vec2 texcoord, in vec3 direct, in vec3 normal){
    return mat.transmission;
}

vec4 fetchNormal(in Submat mat, in vec2 texcoord, in vec3 direct, in vec3 normal){
    vec4 nmap = vec4(0.5f, 0.5f, 1.0f, 1.0f);
    if (validateTexture(mat.bumpPart)) {
        nmap = fetchPart(mat.bumpPart, vec2(texcoord.x, texcoord.y));
    }
    return nmap;
}

vec4 fetchNormal(in Submat mat, in vec2 texcoord, in ivec2 offset, in vec3 direct, in vec3 normal){
    vec4 nmap = vec4(0.5f, 0.5f, 1.0f, 1.0f);
    if (validateTexture(mat.bumpPart)) {
        nmap = fetchPart(mat.bumpPart, vec2(texcoord.x, texcoord.y), offset);
    }
    return nmap;
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
        return normalize(mix(vec3(0.0f, 0.0f, 1.0f), fma(tc, vec3(2.0f), vec3(-1.0f)), vec3(1.0f)));
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


void generateLightPolygon(in vec3 center, in float radius, inout vec3 polygon[4]){
    polygon[0] = center + vec3( 1.0f, 0.0f,  1.0f) * radius;
    polygon[1] = center + vec3(-1.0f, 0.0f,  1.0f) * radius;
    polygon[2] = center + vec3( 1.0f, 0.0f, -1.0f) * radius;
    polygon[3] = center + vec3(-1.0f, 0.0f, -1.0f) * radius;
}

// for polygonal light intersection
float intersectTriangle(in vec3 orig, in vec3 dir, in vec3 ve[3], inout vec2 UV, in bool valid) {
    if (!valid) return INFINITY;

     vec3 e1 = ve[1] - ve[0];
     vec3 e2 = ve[2] - ve[0];

    valid = valid && !(length(e1) < 0.00001f && length(e2) < 0.00001f);
    if (!valid) return INFINITY;

     vec3 pvec = cross(dir, e2);
     float det = dot(e1, pvec);

#ifndef CULLING
    if (abs(det) <= 0.0f) valid = false;
#else
    if (det <= 0.0f) valid = false;
#endif
    if (!valid) return INFINITY;

     vec3 tvec = orig - ve[0];
     float u = dot(tvec, pvec);
     vec3 qvec = cross(tvec, e1);
     float v = dot(dir, qvec);
     vec3 uvt = vec3(u, v, dot(e2, qvec)) / det;

    if (
        any(lessThan(uvt.xy, vec2(0.f))) || 
        any(greaterThan(vec2(uvt.x) + vec2(0.f, uvt.y), vec2(1.f))) 
    ) valid = false;
    if (!valid) return INFINITY;

    UV.xy = uvt.xy;
    return (lessF(uvt.z, 0.0f) || !valid) ? INFINITY : uvt.z;
}



float intersectQuad(in vec3 orig, in vec3 dir, in vec3 ve[4], inout vec2 UV, inout int halfp){
    // test first half
    halfp = 0;
    vec3 vcs[3] = {ve[0], ve[1], ve[2]};
    float first = intersectTriangle(orig, dir, vcs, UV, true);
    if (first < INFINITY) return first;

    // if half of quad failure, intersect with another
    halfp = 1;
    vcs[0] = ve[3], vcs[1] = ve[0];
    float second = intersectTriangle(orig, dir, vcs, UV, true);
    return second;
}





// Linearly Transformed Cosines
///////////////////////////////

float IntegrateEdge(vec3 v1, vec3 v2)
{
    float cosTheta = dot(v1, v2);
    float theta = acos(cosTheta);    
    float res = cross(v1, v2).z * ((theta > 0.001) ? theta/sin(theta) : 1.0);

    return res;
}

void ClipQuadToHorizon(inout vec3 L[5], out int n)
{
    // detect clipping config
    int config = 0;
    if (L[0].z > 0.0) config += 1;
    if (L[1].z > 0.0) config += 2;
    if (L[2].z > 0.0) config += 4;
    if (L[3].z > 0.0) config += 8;

    // clip
    n = 0;

    if (config == 0)
    {
        // clip all
    }
    else if (config == 1) // V1 clip V2 V3 V4
    {
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 2) // V2 clip V1 V3 V4
    {
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 3) // V1 V2 clip V3 V4
    {
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 4) // V3 clip V1 V2 V4
    {
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    }
    else if (config == 5) // V1 V3 clip V2 V4) impossible
    {
        n = 0;
    }
    else if (config == 6) // V2 V3 clip V1 V4
    {
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 7) // V1 V2 V3 clip V4
    {
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 8) // V4 clip V1 V2 V3
    {
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] =  L[3];
    }
    else if (config == 9) // V1 V4 clip V2 V3
    {
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    }
    else if (config == 10) // V2 V4 clip V1 V3) impossible
    {
        n = 0;
    }
    else if (config == 11) // V1 V2 V4 clip V3
    {
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 12) // V3 V4 clip V1 V2
    {
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    }
    else if (config == 13) // V1 V3 V4 clip V2
    {
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    }
    else if (config == 14) // V2 V3 V4 clip V1
    {
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    }
    else if (config == 15) // V1 V2 V3 V4
    {
        n = 4;
    }
    
    if (n == 3)
        L[3] = L[0];
    if (n == 4)
        L[4] = L[0];
}

vec3 LTC_Evaluate(
    vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4], bool twoSided)
{
    // construct orthonormal basis around N
    vec3 T1, T2;
    T1 = normalize(V - N*dot(V, N));
    T2 = cross(N, T1);

    // rotate area light in (T1, T2, N) basis
    Minv = Minv * transpose(mat3(T1, T2, N));

    // polygon (allocate 5 vertices for clipping)
    vec3 L[5];
    L[0] = Minv * (points[0] - P);
    L[1] = Minv * (points[1] - P);
    L[2] = Minv * (points[2] - P);
    L[3] = Minv * (points[3] - P);

    int n;
    ClipQuadToHorizon(L, n);
    
    if (n == 0)
        return vec3(0, 0, 0);

    // project onto sphere
    L[0] = normalize(L[0]);
    L[1] = normalize(L[1]);
    L[2] = normalize(L[2]);
    L[3] = normalize(L[3]);
    L[4] = normalize(L[4]);

    // integrate
    float sum = 0.0;

    sum += IntegrateEdge(L[0], L[1]);
    sum += IntegrateEdge(L[1], L[2]);
    sum += IntegrateEdge(L[2], L[3]);
    if (n >= 4)
        sum += IntegrateEdge(L[3], L[4]);
    if (n == 5)
        sum += IntegrateEdge(L[4], L[0]);

    sum = twoSided ? abs(sum) : max(0.0, sum);

    vec3 Lo_i = vec3(sum, sum, sum);

    return Lo_i;
}





vec3 triangleRandomPoint(in vec3 tri[3]){
     float r1 = random();
     float r2 = random();
    return (1.0f - sqrt(r1)) * tri[0] + sqrt(r1) * (1.0f - r2) * tri[1] + r2 * sqrt(r1) * tri[2];
}






Ray reflection(in Ray newRay, in Hit hit, in vec3 color, in vec3 normal, in float refly){
    if (newRay.params.w == 1) return newRay;
    newRay.direct.xyz = normalize(mix(reflect(newRay.direct.xyz, normal), randomCosine(normal), clamp(refly * random(), 0.0f, 1.0f)));
    //newRay.direct.xyz = normalize(mix(reflect(newRay.direct.xyz, normal), randomCosine(normal), clamp(refly, 0.0f, 1.0f)));
    newRay.color.xyz *= color;
    newRay.params.x = SUNLIGHT_CAUSTICS ? 0 : 1;
    newRay.params.z = 1;
    newRay.bounce = min(3, newRay.bounce); // normal mode
    //newRay.bounce = min(2, newRay.bounce); // easier mode
    newRay.origin.xyz = fma(faceforward(hit.normal.xyz, newRay.direct.xyz, -hit.normal.xyz), vec3(GAP), newRay.origin.xyz); // padding
    return newRay;
}

Ray refraction(in Ray newRay, in Hit hit, in vec3 color, in vec3 normal, in float inior, in float outior, in float glossiness){
     vec3 refrDir = normalize(  refract(newRay.direct.xyz, normal, inior / outior)  );
     bool refrc = equalF(inior, outior);

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


vec3 lightCenter(in int i){
     vec3 playerCenter = vec3(0.0f);
     vec3 lvec = normalize(lightUniform.lightNode[i].lightVector.xyz) * (lightUniform.lightNode[i].lightVector.y < 0.0f ? -1.0f : 1.0f);
    return fma(lvec, vec3(lightUniform.lightNode[i].lightVector.w), (lightUniform.lightNode[i].lightOffset.xyz + playerCenter.xyz));
}

vec3 sLight(in int i){
    return fma(randomDirectionInSphere(), vec3(lightUniform.lightNode[i].lightColor.w), lightCenter(i));
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


float DRo = 1.f;

vec3 extractPivot(vec3 wo, float alpha, out float brdfScale)
{
	// fetch pivot fit params
	float theta = acos(wo.z);
	vec2 fitLookup = vec2(sqrt(alpha), 2.0f * theta / 3.14159f);
	fitLookup = fma(fitLookup, vec2(63.0f / 64.0f), vec2(0.5f / 64.0f));
	vec4 pivotParams = texture(u_PivotSampler, fitLookup);
	float pivotNorm = pivotParams.r;
	float pivotElev = pivotParams.g;
	vec3 pivot = pivotNorm * vec3(sin(pivotElev), 0, cos(pivotElev));

	// express the pivot in tangent space
	mat3 basis;
	basis[0] = wo.z < 0.999 ? normalize(wo - vec3(0, 0, wo.z)) : vec3(1, 0, 0);
	basis[1] = cross(vec3(0, 0, 1), basis[0]);
	basis[2] = vec3(0, 0, 1);
	pivot = basis * pivot;

	// return
	brdfScale = pivotParams.a;
	return pivot;
}

const uint u_SamplesPerPass = 4;
mat3 tbn_light = mat3(1.0f);
vec3 dirl = vec3(0.0f);

/*
Ray directLightRoughness(in int i, in Ray directRay, in Hit hit, in vec3 color, in vec3 normal){
    if (directRay.params.w == 1) return directRay;
    directRay.bounce = min(1, directRay.bounce);
    directRay.actived = 1;
    directRay.params.w = 1;
    directRay.params.x = 0;
    directRay.params.y = i;

    // extract attributes
	vec3 wo = normalize(dirl);//reflect(directRay.direct.xyz, normal);//-normalize(directRay.direct.xyz);
	mat3 tg = tbn_light;
    wo = normalize(tg * -wo);
    
    // fetch pivot fit params
	float brdfScale = 0.0f; // this won't be used here
    float alpha = clamp(DRo * 0.5f, 0.0f, 1.0f);//clamp(pow(DRo * 0.5f, 2.0f), 0.0f, 1.0f);

	vec3 pivot = extractPivot(wo, alpha, brdfScale);
    vec3 Li = vec3(1);
    vec3 Lo = vec3(0);

	// iterate over all spheres
    vec3 spherePos = tg * (lightCenter(i).xyz - directRay.origin.xyz);
    float sphereRadius = lightUniform.lightNode[i].lightColor.w;
    sphere s = sphere(spherePos, sphereRadius);
    float invSphereMagSqr = 1.0f / dot(s.pos, s.pos);
    vec3 capDir = s.pos * sqrt(invSphereMagSqr);
    float capCos = sqrt(1.0 - s.r * s.r * invSphereMagSqr);

    cap c = cap(capDir, capCos);
    cap c_std = cap_to_pcap(c, pivot);

    if (c.z < 0.99) {
        // Joint MIS: loop over all samples
        for (int j = 0; j < u_SamplesPerPass; ++j) {
            vec2 u2 = vec2(random(), random());

            // importance sample the BRDF
            if (true) {
                vec3 wm = ggx_sample(u2, wo, alpha);
                vec3 wi = 2.0 * wm * dot(wo, wm) - wo;
                float pdf1;
                float frp = ggx_evalp(wi, wo, alpha, pdf1);
                float raySphereIntersection = pdf_cap(wi, c);

                // raytrace the sphere light
                if (pdf1 > 0.0 && raySphereIntersection > 0.0) {
                    float pdf2 = pdf_pcap_fast(wi, c_std, pivot);
                    float misWeight = pdf1 * pdf1;
                    float misNrm = pdf1 * pdf1 + pdf2 * pdf2;

                    Lo+= Li * frp / pdf1 * misWeight / misNrm;
                }
            }

            // importance sample the pivot transformed spherical cap
            if (true) {
                vec3 wi = u2_to_pcap(u2, c_std, pivot);
                float pdf1;
                float frp = ggx_evalp(wi, wo, alpha, pdf1);
                float pdf2 = pdf_pcap_fast(wi, c_std, pivot);

                if (pdf2 > 0.0) {
                    float misWeight = pdf2 * pdf2;
                    float misNrm = pdf1 * pdf1 + pdf2 * pdf2;

                    Lo+= Li * frp / pdf2 * misWeight / misNrm;
                }
            }
        }
    } else {
        // classic MIS: loop over all samples
        for (int j = 0; j < u_SamplesPerPass; ++j) {
            vec2 u2 = vec2(random(), random());

            // importance sample the BRDF
            if (true) {
                vec3 wm = ggx_sample(u2, wo, alpha);
                vec3 wi = 2.0 * wm * dot(wo, wm) - wo;
                float pdf1;
                float frp = ggx_evalp(wi, wo, alpha, pdf1);
                float raySphereIntersection = pdf_cap(wi, c);

                // raytrace the sphere light
                if (pdf1 > 0.0 && raySphereIntersection > 0.0) {
                    float pdf2 = pdf_cap(wi, c);
                    float misWeight = pdf1 * pdf1;
                    float misNrm = pdf1 * pdf1 + pdf2 * pdf2;

                    Lo+= Li * frp / pdf1 * misWeight / misNrm;
                }
            }

            // importance sample the spherical cap
            if (true) {
                vec3 wi = u2_to_cap(u2, c);
                float pdf1;
                float frp = ggx_evalp(wi, wo, alpha, pdf1);
                float pdf2 = pdf_cap(wi, c);

                if (pdf2 > 0.0) {
                    float misWeight = pdf2 * pdf2;
                    float misNrm = pdf1 * pdf1 + pdf2 * pdf2;

                    Lo+= Li * frp / pdf2 * misWeight / misNrm;
                }
            }
        }
    }

    vec3 ltr = lightCenter(i).xyz-directRay.origin.xyz;
    vec3 ldirect = normalize(sLight(i) - directRay.origin.xyz);
    float diffuseWeight = clamp(dot(ldirect, normal), 0.0f, 1.0f);

    directRay.direct.xyz = ldirect;
    directRay.color.xyz *= color * clamp(Lo.xyz / float(u_SamplesPerPass), 0.0f, 1.0f);
    return directRay;
}

Ray directLight(in int i, in Ray directRay, in Hit hit, in vec3 color, in vec3 normal, in float roughness){
    DRo = clamp(roughness, 0.0001f, 1.0f);
    Ray drtRay = directLightRoughness(i, directRay, hit, color, normal);
    DRo = 1.f;
    return drtRay;
}
*/

Ray directLight(in int i, in Ray directRay, in Hit hit, in vec3 color, in vec3 normal){
    if (directRay.params.w == 1) return directRay;
    directRay.bounce = min(1, directRay.bounce);
    directRay.actived = 1;
    directRay.params.w = 1;
    directRay.params.x = 0;
    directRay.params.y = i;

    vec3 ltr = lightCenter(i).xyz-directRay.origin.xyz;
    vec3 ldirect = normalize(sLight(i) - directRay.origin.xyz);
    float cos_a_max = sqrt(1.f - clamp(lightUniform.lightNode[i].lightColor.w * lightUniform.lightNode[i].lightColor.w / sqlen(ltr), 0.0f, 1.0f));
    float diffuseWeight = clamp(dot(ldirect, normal), 0.0f, 1.0f);

    directRay.direct.xyz = ldirect;
    directRay.color.xyz *= color * clamp(diffuseWeight * ((1.0f - cos_a_max) * 2.0f), 0.0f, 1.0f);
    if (DRo < 0.9999f) directRay.color.xyz *= vec3(0.0f);
    return directRay;
    
    //DRo = 1.f;
    //return directLightRoughness(i, directRay, hit, color, normal);
}








Ray diffuse(in Ray newRay, in Hit hit, in vec3 color, in vec3 normal){
    if (newRay.params.w == 1) return newRay;
    newRay.color.xyz *= color;
    newRay.direct.xyz = normalize(randomCosine(normal));
    //newRay.direct.xyz = normalize(mix(reflect(newRay.direct.xyz, normal), randomCosine(normal), clamp(random(), 0.0f, 1.0f)));
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
    return createRay(directRay);
}

vec3 lightCenterSky(in int i) {
     vec3 playerCenter = vec3(0.0f);
     vec3 lvec = normalize(lightUniform.lightNode[i].lightVector.xyz) * 1000.0f;
    return lightUniform.lightNode[i].lightOffset.xyz + lvec + playerCenter.xyz;
}

float intersectSphere(in vec3 origin, in vec3 ray, in vec3 sphereCenter, in float sphereRadius) {
     vec3 toSphere = origin - sphereCenter;
     float a = dot(ray, ray);
     float b = 2.0f * dot(toSphere, ray);
     float c = dot(toSphere, toSphere) - sphereRadius*sphereRadius;
     float discriminant = fma(b,b,-4.0f*a*c);
    if(discriminant > 0.0f) {
         float da = 0.5f / a;
         float t1 = (-b - sqrt(discriminant)) * da;
         float t2 = (-b + sqrt(discriminant)) * da;
         float mn = min(t1, t2);
         float mx = max(t1, t2);
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
    return lightUniform.lightNode[lc].lightColor.xyz;
}

#endif
