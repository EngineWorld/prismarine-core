
cd %~dp0
set CFLAGS=--target-env=opengl -x glsl -Werror -DUSE_OPENGL 
set HFLAGS=--target-env=vulkan -x hlsl -Werror 
set INDIR=.\
set DXDIR=..\directx-sdk\
set OUTDIR=..\build\shaders-spv\
set OUTSHR=..\build\shaders\
set VXL2=tools\
set RNDR=render\
set EXPR=experimental\
set HLBV=hlbvh\
set RDXI=radix\

set CMPPROF=-fshader-stage=compute
set FRGPROF=-fshader-stage=fragment
set VRTPROF=-fshader-stage=vertex
set GMTPROF=-fshader-stage=geometry


set CMPPROFM=-e CSMain -S compute --hlsl-iomap --target-env opengl -V -D 


mkdir %OUTDIR%
mkdir %OUTDIR%%VXL2%
mkdir %OUTDIR%%RNDR%
mkdir %OUTDIR%%HLBV%
mkdir %OUTDIR%%RDXI%
mkdir %OUTDIR%%HLBV%next-gen-sort

call glslc %CFLAGS% %CMPPROF% %INDIR%%VXL2%loader.comp        -o %OUTDIR%%VXL2%loader.comp.spv -DINVERT_TX_Y
call glslc %CFLAGS% %CMPPROF% %INDIR%%VXL2%loader.comp        -o %OUTDIR%%VXL2%loader-int16.comp.spv -DINVERT_TX_Y -DENABLE_INT16_LOADING
call glslc %CFLAGS% %FRGPROF% %INDIR%%RNDR%render.frag        -o %OUTDIR%%RNDR%render.frag.spv
call glslc %CFLAGS% %VRTPROF% %INDIR%%RNDR%render.vert        -o %OUTDIR%%RNDR%render.vert.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%begin.comp         -o %OUTDIR%%RNDR%begin.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%camera.comp        -o %OUTDIR%%RNDR%camera.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%clear.comp         -o %OUTDIR%%RNDR%clear.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%reclaim.comp       -o %OUTDIR%%RNDR%reclaim.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%sampler.comp       -o %OUTDIR%%RNDR%sampler.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%testmat.comp       -o %OUTDIR%%RNDR%testmat.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%traverse.comp      -o %OUTDIR%%RNDR%traverse.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%resolver.comp      -o %OUTDIR%%RNDR%resolver.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%directTraverse.comp  -o %OUTDIR%%RNDR%directTraverse.comp.spv

call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%aabbmaker.comp     -o %OUTDIR%%HLBV%aabbmaker.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%build.comp         -o %OUTDIR%%HLBV%build.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%minmax.comp        -o %OUTDIR%%HLBV%minmax.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%refit.comp         -o %OUTDIR%%HLBV%refit.comp.spv

call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%histogram.comp     -o %OUTDIR%%RDXI%histogram.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%permute.comp       -o %OUTDIR%%RDXI%permute.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%prefix-scan.comp   -o %OUTDIR%%RDXI%prefix-scan.comp.spv

set OPTFLAGS= ^
--unify-const ^
--flatten-decorations ^
--convert-local-access-chains ^
--fold-spec-const-op-composite ^
--merge-blocks ^
--inline-entry-points-exhaustive ^
--eliminate-dead-code-aggressive ^
--eliminate-insert-extract ^
--eliminate-common-uniform ^
--eliminate-dead-branches ^
--eliminate-dead-const ^
--eliminate-local-single-block ^
--eliminate-local-single-store ^
--eliminate-local-multi-store


call spirv-opt %OPTFLAGS% %OUTDIR%%VXL2%loader.comp.spv         -o %OUTDIR%%VXL2%loader.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%VXL2%loader-int16.comp.spv   -o %OUTDIR%%VXL2%loader-int16.comp.spv

call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%directTraverse.comp.spv -o %OUTDIR%%RNDR%directTraverse.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%resolver.comp.spv       -o %OUTDIR%%RNDR%resolver.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%traverse.comp.spv       -o %OUTDIR%%RNDR%traverse.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%reclaim.comp.spv        -o %OUTDIR%%RNDR%reclaim.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%begin.comp.spv          -o %OUTDIR%%RNDR%begin.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%testmat.comp.spv        -o %OUTDIR%%RNDR%testmat.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%camera.comp.spv         -o %OUTDIR%%RNDR%camera.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%sampler.comp.spv        -o %OUTDIR%%RNDR%sampler.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%clear.comp.spv          -o %OUTDIR%%RNDR%clear.comp.spv

call spirv-opt %OPTFLAGS% %OUTDIR%%HLBV%aabbmaker.comp.spv      -o %OUTDIR%%HLBV%aabbmaker.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%HLBV%build.comp.spv          -o %OUTDIR%%HLBV%build.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%HLBV%minmax.comp.spv         -o %OUTDIR%%HLBV%minmax.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%HLBV%refit.comp.spv          -o %OUTDIR%%HLBV%refit.comp.spv

call spirv-opt %OPTFLAGS% %OUTDIR%%RDXI%histogram.comp.spv      -o %OUTDIR%%RDXI%histogram.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RDXI%permute.comp.spv        -o %OUTDIR%%RDXI%permute.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RDXI%prefix-scan.comp.spv    -o %OUTDIR%%RDXI%prefix-scan.comp.spv

pause
