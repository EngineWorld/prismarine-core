#ifndef _MORTON_H
#define _MORTON_H

#ifdef INT64_MORTON

uint64_t part1By2_64(in uint a){
    uint64_t x = uint64_t(a);
    x &= 0x1fffffl;
    x = (x | (x << 32)) & 0x1f00000000ffffl;
    x = (x | (x << 16)) & 0x1f0000ff0000ffl;
    x = (x | (x << 8 )) & 0x100f00f00f00f00fl;
    x = (x | (x << 4 )) & 0x10c30c30c30c30c3l;
    x = (x | (x << 2 )) & 0x1249249249249249l;
    return x;
}

uint64_t encodeMorton3_64(in uvec3 a)
{
    return part1By2_64(a.x) | (part1By2_64(a.y) << 1) | (part1By2_64(a.z) << 2);
}

uint compact1By2_64(in uint64_t a)
{
    uint64_t x = a;
    x &= 0x1249249249249249l;
    x = (x | (x >>  2)) & 0x10c30c30c30c30c3l;
    x = (x | (x >>  4)) & 0x100f00f00f00f00fl;
    x = (x | (x >>  8)) & 0x1f0000ff0000ffl;
    x = (x | (x >> 16)) & 0x1f00000000ffffl;
    x = (x | (x >> 32)) & 0x1fffffl;
    return uint(x);
}

uvec3 decodeMorton3_64(in uint64_t code)
{
    return uvec3(compact1By2_64(code >> 0), compact1By2_64(code >> 1), compact1By2_64(code >> 2));
}

#else

uint part1By2_64(in uint a){
    uint x = a;
    x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0x030000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}


uint part1By5(in uint x){
    uint answer = 0;
    for (uint i = 0; i < 32 / 6; ++i) {
        answer |= ((x & (1u << i)) << 5u*i);
    }
    return answer;
}

uint encodeMorton3_64(in uvec3 a, in uvec3 b)
{
    return 
        (part1By5(a.x) << 0) | (part1By5(a.y) << 1) | (part1By5(a.z) << 2) | 
        (part1By5(b.x) << 3) | (part1By5(b.y) << 4) | (part1By5(b.z) << 5);
}




uint MortonToHilbert3D( const uint morton, const uint bits )
{
    uint hilbert = morton;
    if( bits > 1 )
    {
    uint block = ( ( bits * 3 ) - 3 );
    uint hcode = ( ( hilbert >> block ) & 7 );
    uint mcode, shift, signs;
    shift = signs = 0;
    while( block > 0 )
    {
        block -= 3;
        hcode <<= 2;
        mcode = ( ( 0x20212021 >> hcode ) & 3 );
        shift = ( ( 0x48 >> ( 7 - shift - mcode ) ) & 3 );
        signs = ( ( signs | ( signs << 3 ) ) >> mcode );
        signs = ( ( signs ^ ( 0x53560300 >> hcode ) ) & 7 );
        mcode = ( ( hilbert >> block ) & 7 );
        hcode = mcode;
        hcode = ( ( ( hcode | ( hcode << 3 ) ) >> shift ) & 7 );
        hcode ^= signs;
        hilbert ^= ( ( mcode ^ hcode ) << block );
    }
    }
    hilbert ^= ( ( hilbert >> 1 ) & 0x92492492 );
    hilbert ^= ( ( hilbert & 0x92492492 ) >> 1 );
    return( hilbert );
}



uint encodeMorton3_64(in uvec3 a)
{
    return MortonToHilbert3D(part1By2_64(a.x) | (part1By2_64(a.y) << 1) | (part1By2_64(a.z) << 2), 10);
    //return part1By2_64(a.x) | (part1By2_64(a.y) << 1) | (part1By2_64(a.z) << 2);
}

uint compact1By2_64(in uint a)
{
    uint x = a;
    x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >>  8)) & 0x030000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    return x;
}

uvec3 decodeMorton3_64(in uint code)
{
    return uvec3(compact1By2_64(code >> 0), compact1By2_64(code >> 1), compact1By2_64(code >> 2));
}

#endif

#endif
