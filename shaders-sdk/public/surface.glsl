
vec4 fetchSpecular(in Material mat, in vec2 texcoord){
    vec4 specular = mat.specular;
    if (validateTexture(mat.specularPart)) {
        specular = fetchTexture(mat.specularPart, texcoord);
    }
    return specular;
}

vec4 fetchEmissive(in Material mat, in vec2 texcoord){
    vec4 emission = vec4(0.0f);
    if (validateTexture(mat.emissivePart)) {
        emission = fetchTexture(mat.emissivePart, texcoord);
    }
    return emission;
}

vec4 fetchTransmission(in Material mat, in vec2 texcoord){
    return mat.transmission;
}

vec4 fetchNormal(in Material mat, in vec2 texcoord){
    vec4 nmap = vec4(0.5f, 0.5f, 1.0f, 1.0f);
    if (validateTexture(mat.bumpPart)) {
        nmap = fetchTexture(mat.bumpPart, vec2(texcoord.x, texcoord.y));
    }
    return nmap;
}

vec4 fetchNormal(in Material mat, in vec2 texcoord, in ivec2 offset){
    vec4 nmap = vec4(0.5f, 0.5f, 1.0f, 1.0f);
    if (validateTexture(mat.bumpPart)) {
        nmap = fetchTexture(mat.bumpPart, vec2(texcoord.x, texcoord.y), offset);
    }
    return nmap;
}

vec3 getNormalMapping(in Material mat, vec2 texcoordi) {
    vec3 tc = fetchNormal(mat, texcoordi).xyz;
    if(equalF(tc.x, tc.y) && equalF(tc.x, tc.z)){
        const ivec3 off = ivec3(0,0,1);
        const float size = 1.0f;
        const float pike = 2.0f;
        vec3 p00 = vec3(0.0f, 0.0f, fetchNormal(mat, texcoordi, off.yy).x * pike);
        vec3 p01 = vec3(size, 0.0f, fetchNormal(mat, texcoordi, off.zy).x * pike);
        vec3 p10 = vec3(0.0f, size, fetchNormal(mat, texcoordi, off.yz).x * pike);
        return normalize(cross(p10 - p00, p01 - p00));
    } else {
        return normalize(mix(vec3(0.0f, 0.0f, 1.0f), fma(tc, vec3(2.0f), vec3(-1.0f)), vec3(1.0f)));
    }
    return vec3(0.0, 0.0, 1.0);
}

vec4 fetchDiffuse(in Material mat, in vec2 texcoord){
    vec4 result = vec4(0.0f);
    result = mat.diffuse;
    if (validateTexture(mat.diffusePart)) {
        result = fetchTexture(mat.diffusePart, texcoord);
    }
    return result;
}

void surfaceShade(inout SurfaceData surface, in Material material, in Hit hit, in Ray ray){
    vec4 specular = fetchSpecular(material, hit.texcoord.xy);
    surface.albedo = fetchDiffuse(material, hit.texcoord.xy);
    surface.specular = vec4(1.f);
    surface.normal = getNormalMapping(material, hit.texcoord.xy);
    surface.emission = fetchEmissive(material, hit.texcoord.xy);
    surface.roughness = specular.y;
    surface.metallic = specular.z;
    surface.culling = 0; // not culling
}

#define SurfaceShader surfaceShade
