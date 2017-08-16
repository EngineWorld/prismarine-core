#ifndef _RANDOM_H
#define _RANDOM_H

uint randomClocks = 0;
uint globalInvocationSMP = 0;

uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

float floatConstruct( in uint m ) {
     uint ieeeMantissa = 0x007FFFFFu;          // binary32 mantissa bitmask
     uint ieeeOne      = 0x3F800000u;          // 1.0 in IEEE binary32
    m &= ieeeMantissa;                         // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                              // Add fractional part to 1.0
    return fract(uintBitsToFloat(m)); // Range [0:1]
}

float random( in uint   x ) { return floatConstruct(hash(x)); }
float random( in uvec2  v ) { return floatConstruct(hash(v)); }
float random( in uvec3  v ) { return floatConstruct(hash(v)); }
float random( in uvec4  v ) { return floatConstruct(hash(v)); }

float random() {
#ifdef USE_ARB_CLOCK
    return random(uvec3( globalInvocationSMP, clock2x32ARB()));
#else
    //uint hs = ++randomClocks;
    uint hs = randomClocks; randomClocks = hash(++randomClocks);
    //uint hs = hash(++randomClocks); randomClocks = hs;
    return random(uvec3( globalInvocationSMP, hs, RAY_BLOCK materialUniform.time << 5 ));
#endif
}

vec3 randomCosine(in vec3 normal) {
     float up = sqrt(random());
     float over = sqrt(1.f - up * up);
     float around = random() * TWO_PI;

    vec3 perpendicular0 = vec3(0, 0, 1);
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        perpendicular0 = vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        perpendicular0 = vec3(0, 1, 0);
    }

     vec3 perpendicular1 = normalize( cross(normal, perpendicular0) );
     vec3 perpendicular2 =            cross(normal, perpendicular1);
    return normalize(
        fma(normal, vec3(up),
            fma( perpendicular1 , vec3(cos(around)) * over,
                 perpendicular2 * vec3(sin(around)) * over
            )
        )
    );
}

vec3 randomDirectionInSphere() {
     float up = fma(random(), 2.0f, -1.0f);
     float over = sqrt(1.f - up * up);
     float around = random() * TWO_PI;
    return normalize(vec3( up, cos(around) * over, sin(around) * over ));
}

#endif
