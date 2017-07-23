set FLAGS= -E CSMain -T cs_5_0 -HV 2017 -WX -O0 
 dxc %FLAGS% render/intersection.compute  -Fo bin/render/intersection.bin
 dxc %FLAGS% hlbvh/aabbmaker.compute      -Fo bin/hlbvh/aabbmaker.bin
 dxc %FLAGS% hlbvh/build.compute          -Fo bin/hlbvh/build.bin
 dxc %FLAGS% hlbvh/minmax.compute         -Fo bin/hlbvh/minmax.bin
 dxc %FLAGS% hlbvh/refit.compute          -Fo bin/hlbvh/refit.bin
 dxc %FLAGS% radix/histogram.compute      -Fo bin/radix/histogram.bin
 dxc %FLAGS% radix/permute.compute        -Fo bin/radix/permute.bin
 dxc %FLAGS% radix/prefix-scan.compute    -Fo bin/radix/prefix-scan.bin
 
pause
