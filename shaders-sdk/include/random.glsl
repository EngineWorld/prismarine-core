#ifndef _RANDOM_H
#define _RANDOM_H

float counter = 0.0f;
int globalInvocationSMP = 0;

int hash( in int x ) {
    x += ( x << 10 );
    x ^= ( x >>  6 );
    x += ( x <<  3 );
    x ^= ( x >> 11 );
    x += ( x << 15 );
    return x;
}

int hash( in ivec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
int hash( in ivec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
int hash( in ivec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

float floatConstruct( in int m ) {
    const int ieeeMantissa = 0x007FFFFF; // binary32 mantissa bitmask
    const int ieeeOne      = 0x3F800000; // 1.0 in IEEE binary32
    m &= ieeeMantissa;                   // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                        // Add fractional part to 1.0
    return fract(uintBitsToFloat( m ));  // Range [0:1]
}

float random( in float x ) { return floatConstruct(hash(floatBitsToInt(x))); }
float random( in vec2  v ) { return floatConstruct(hash(floatBitsToInt(v))); }
float random( in vec3  v ) { return floatConstruct(hash(floatBitsToInt(v))); }
float random( in vec4  v ) { return floatConstruct(hash(floatBitsToInt(v))); }

float random() {
    const float x = float(globalInvocationSMP);
    const float y = counter + RAY_BLOCK randomUniform.time;
    counter = random(vec2(x, y));
    return counter;
}

vec3 randomCosine(in vec3 normal) {
    const float up = sqrt(random());
    const float over = sqrt(-fma(up, up, -1.0f));
    const float around = random() * TWO_PI;

    vec3 perpendicular0 = vec3(0, 0, 1);
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        perpendicular0 = vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        perpendicular0 = vec3(0, 1, 0);
    }

    const vec3 perpendicular1 = normalize( cross(normal, perpendicular0) );
    const vec3 perpendicular2 =            cross(normal, perpendicular1);
    return normalize(
        fma(normal, vec3(up),
            fma( perpendicular1 , vec3(cos(around)) * over,
                 perpendicular2 * vec3(sin(around)) * over
            )
        )
    );
}

vec3 randomDirectionInSphere() {
    const float up = fma(random(), 2.0f, -1.0f);
    const float over = sqrt(fma(up, -up, 1.0f));
    const float around = random() * TWO_PI;
    return normalize(vec3( up, cos(around) * over, sin(around) * over ));
}

#endif
