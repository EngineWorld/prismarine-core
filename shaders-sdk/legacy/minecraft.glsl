
    float grefl = 0.2f + pow((1.0f - abs(dot(normal, newRay.direct.xyz))), 2.0f) * 0.6f;
    const float glossiness = clamp(1.0f / log2(materialCache.reflectivity), 0.0f, 1.0f);
    const bool isWater = iseq16(hit.vmods.x, 8.f) || iseq16(hit.vmods.x, 9.f);
    const bool isGlass = iseq16(hit.vmods.x, 20.f) || iseq16(hit.vmods.x, 102.f);
    const bool isStained = iseq16(hit.vmods.x, 95.f) || iseq16(hit.vmods.x, 160.f);



    if ( iseq16(hit.vmods.x, 89.f) ) { // glowstone
        emi = vec4(8.0f);
    }

    if ( iseq16(hit.vmods.x, 10.f) || iseq16(hit.vmods.x, 11.f) ) { // lava
        emi = vec4(7.0f);
    }

    if ( iseq16(hit.vmods.x, 76.f) ) { // redstone torch
        emi = vec4(8.0f);
    }

    if ( iseq16(hit.vmods.x, 50.f) ) { // torch
        emi = vec4(12.0f);
    }




    if (isGlass) {

        // glass transparency
        float trnsy = prom * (1.0f - grefl);
        if (greaterF(trnsy, 0.0f)) emitRay(promised(newRay, hit, normal), hit, normal, trnsy);

        // glass reflection (on transparency)
        float refly = prom * grefl;
        if (greaterF(refly, 0.0f)) emitRay(reflection(newRay, hit, vec3(1.0f), normal, 0.0f), hit, normal, refly);

        {
            // glass frames (diffuse)
            newRay.color.xyz *= 1.0f - prom;
            newRay = diffuse(newRay, hit, tx.xyz, normal);

#ifdef DIRECT_LIGHT
            if (diffused) applyLight(directLight(0, newRay, hit, vec3(1.0f), normal), newRay, surfacenormal);
#endif

            diffused = true;
        }

    } else

    if (isStained) {

        // glass reflection
        emitRay(reflection(newRay, hit, vec3(1.0f), normal, 0.0f), hit, normal, grefl);
        newRay.color.xyz *= (1.0f - grefl);

        // stained transparency (with multiply)
        newRay = refraction(newRay, hit, mix(tx.xyz, vec3(1.0f), vec3(prom)), normal, 1.0f, 1.0f, 0.0f);

    } else

    if (isWater) {
        const float frameTimeCounter = float(double(RAY_BLOCK materialUniform.iModifiers0.w) / 60000.0f) * 4.0f;
        vec3 posxz = (inverse(RAY_BLOCK materialUniform.transformModifier) * vec4(newRay.origin.xyz, 1.0f)).xyz;
        posxz.x += sin(posxz.z+frameTimeCounter)*0.25;
        posxz.z += cos(posxz.x+frameTimeCounter*0.5)*0.25;

        const float deltaPos = 0.1f;
        const float h0 = waterH(posxz, frameTimeCounter);
        const float h1 = waterH(posxz + vec3(deltaPos,0.0,0.0), frameTimeCounter);
        const float h2 = waterH(posxz + vec3(-deltaPos,0.0,0.0), frameTimeCounter);
        const float h3 = waterH(posxz + vec3(0.0,0.0,deltaPos), frameTimeCounter);
        const float h4 = waterH(posxz + vec3(0.0,0.0,-deltaPos), frameTimeCounter);
        const float xDelta = ((h1-h0)+(h0-h2))/deltaPos;
        const float yDelta = ((h3-h0)+(h0-h4))/deltaPos;

        vec3 bump = normalize(vec3(xDelta,yDelta,1.0-xDelta*xDelta-yDelta*yDelta));
        const float bumpmult = 0.1f;
        bump = bump * vec3(bumpmult, bumpmult, bumpmult) + vec3(0.0f, 0.0f, 1.0f - bumpmult);
        normal = normalize(tbn * bump);

        inior = 1.0f;
        outior = 1.333333333f;
        const float inside = dot(prenormal, dir);
        if (inside >= 0.0f) {
            const float tmp = inior;
            inior = outior;
            outior = tmp;
        }

        Ray promRay = reflection(newRay, hit, vec3(1.0f), normal, 0.0f);
        Ray refrRay = refraction(newRay, hit, vec3( 1.0f ), normal, inior, outior, 0.0f);
        const float fres = computeFresnel(normal, newRay.direct.xyz, inior, outior) * (inside <= 0.0f ? 1.0f : 0.0f);
        if (newRay.params.w == 0) {
            emitRay(promRay, hit, normal, fres);
        }
        newRay = refrRay;
        newRay.color.xyz *= 1.0f - fres;

    } else
