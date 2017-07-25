set FLAGS= -E CSMain -T comp -Vout GLSL450 -Vin HLSL6 --explicit-bind ON --wrapper ON
 
::xsc -o "../build/shaders/render/intersection.comp" "./render/intersection.cginc"

 xsc -o ../build/shaders/render/intersection.comp %FLAGS% render/intersection.compute 
 xsc -o ../build/shaders/hlbvh/aabbmaker.comp     %FLAGS% hlbvh/aabbmaker.compute     
 xsc -o ../build/shaders/hlbvh/build.comp         %FLAGS% hlbvh/build.compute         
 xsc -o ../build/shaders/hlbvh/minmax.comp        %FLAGS% hlbvh/minmax.compute        
 xsc -o ../build/shaders/hlbvh/refit.comp         %FLAGS% hlbvh/refit.compute         
 xsc -o ../build/shaders/radix/histogram.comp     %FLAGS% radix/histogram.compute     
 xsc -o ../build/shaders/radix/permute.comp       %FLAGS% radix/permute.compute       
 xsc -o ../build/shaders/radix/prefix-scan.comp   %FLAGS% radix/prefix-scan.compute    

pause