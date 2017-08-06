#ifndef _MORTON_H
#define _MORTON_H

#if (defined(INT64_MORTON) && defined(USE_INT64))

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

#else

uint MortonToHilbert3D( const uint morton, const uint bits ) {
    uint hilbert = morton;
    if ( bits > 1 ) {
        uint block = ( ( bits * 3 ) - 3 );
        uint hcode = ( ( hilbert >> block ) & 7 );
        uint mcode, shift, signs;
        shift = signs = 0;
        while ( block > 0 ) {
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

uint part1By2_64(in uint a){
    uint x = a;
    x &= 0x000003ff;
    x = (x ^ (x << 16)) & 0x030000ff;
    x = (x ^ (x <<  8)) & 0x0300f00f;
    x = (x ^ (x <<  4)) & 0x030c30c3;
    x = (x ^ (x <<  2)) & 0x09249249;
    return x;
}

uint encodeMorton3_64(in uvec3 a) {
    return part1By2_64(a.x) | (part1By2_64(a.y) << 1) | (part1By2_64(a.z) << 2);
}

#endif

#endif
