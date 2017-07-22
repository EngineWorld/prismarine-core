set FLAGS= -E main -T cs_6_0 -HV 2017 -WX -O0 
 dxc %FLAGS% intersection.hlsl       -Fo intersection.bin
 dxc %FLAGS% hlbvh/aabbmaker.hlsl    -Fo hlbvh/aabbmaker.bin
 dxc %FLAGS% hlbvh/build.hlsl        -Fo hlbvh/build.bin
 dxc %FLAGS% hlbvh/minmax.hlsl       -Fo hlbvh/minmax.bin
 dxc %FLAGS% hlbvh/refit.hlsl        -Fo hlbvh/refit.bin
 dxc %FLAGS% radix/histogram.hlsl    -Fo radix/histogram.bin
 dxc %FLAGS% radix/permute.hlsl      -Fo radix/permute.bin
 dxc %FLAGS% radix/prefix-scan.hlsl  -Fo radix/prefix-scan.bin
pause
