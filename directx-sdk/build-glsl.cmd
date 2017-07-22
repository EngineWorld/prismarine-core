set FLAGS= -E CSMain -T comp -Vout GLSL450 -Vin HLSL6 --explicit-bind ON --wrapper ON
 
::xsc -o "../build/shaders/render/intersection.comp" "./render/intersection.hlsl"

 xsc -o ../build/shaders/render/intersection.comp %FLAGS% render/intersection.hlsl 
 xsc -o ../build/shaders/hlbvh/aabbmaker.comp     %FLAGS% hlbvh/aabbmaker.hlsl     
 xsc -o ../build/shaders/hlbvh/build.comp         %FLAGS% hlbvh/build.hlsl         
 xsc -o ../build/shaders/hlbvh/minmax.comp        %FLAGS% hlbvh/minmax.hlsl        
 xsc -o ../build/shaders/hlbvh/refit.comp         %FLAGS% hlbvh/refit.hlsl         
 xsc -o ../build/shaders/radix/histogram.comp     %FLAGS% radix/histogram.hlsl     
 xsc -o ../build/shaders/radix/permute.comp       %FLAGS% radix/permute.hlsl       
 xsc -o ../build/shaders/radix/prefix-scan.comp   %FLAGS% radix/prefix-scan.hlsl    

pause