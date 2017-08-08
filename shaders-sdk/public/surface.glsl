vec4 fetchDiffuse(in Submat mat, in vec2 texcoord){
    vec4 result = vec4(0.0f);
    result = mat.diffuse;
    if (validateTexture(mat.diffusePart)) {
        result = fetchPart(mat.diffusePart, texcoord);
    }
    return result;
}

void surfaceShade(in Submat material, in Hit hit, in Ray ray, inout SurfaceData surface){
    vec4 specular = fetchSpecular(material, hit.texcoord.xy);
    surface.albedo = fetchDiffuse(material, hit.texcoord.xy);
    surface.specular = vec4(1.f);
    surface.normal = getNormalMapping(material, hit.texcoord.xy);
    surface.emission = fetchEmissive(material, hit.texcoord.xy);
    surface.emission.xyz *= 2.0f;
    surface.roughness = specular.y;
    surface.metallic = specular.z;
    surface.culling = 0; // not culling
}

#define SurfaceShader surfaceShade
