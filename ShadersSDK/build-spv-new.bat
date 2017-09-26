:: It is helper for compilation shaders to SPIR-V

cd %~dp0
set CFLAGS=--target-env=opengl -x glsl -Werror -DUSE_OPENGL -E
::set CFLAGS=--target-env=opengl -x glsl -Werror -DUSE_OPENGL -E -DENABLE_AMD_INSTRUCTION_SET -DAMD_F16_BVH 
set INDIR=.\
set OUTDIR=..\Build\shaders-spv\
set OUTSHR=..\Build\shaders\
set VRTX=vertex\
set RNDR=raytracing\
set EXPR=experimental\
set HLBV=hlbvh\
set RDXI=radix\

set CMPPROF=-fshader-stage=compute
set FRGPROF=-fshader-stage=fragment
set VRTPROF=-fshader-stage=vertex
set GMTPROF=-fshader-stage=geometry


set CMPPROFM=-e CSMain -S compute --hlsl-iomap --target-env opengl -V -D 


mkdir %OUTDIR%
mkdir %OUTDIR%%VRTX%
mkdir %OUTDIR%%RNDR%
mkdir %OUTDIR%%HLBV%
mkdir %OUTDIR%%RDXI%
mkdir %OUTDIR%%HLBV%next-gen-sort

call glslc %CFLAGS% %CMPPROF% %INDIR%%VRTX%loader.comp        -o %OUTDIR%%VRTX%loader.comp.spv -DINVERT_TX_Y
call glslc %CFLAGS% %CMPPROF% %INDIR%%VRTX%loader.comp        -o %OUTDIR%%VRTX%loader-int16.comp.spv -DINVERT_TX_Y -DENABLE_INT16_LOADING
call glslc %CFLAGS% %FRGPROF% %INDIR%%RNDR%render.frag        -o %OUTDIR%%RNDR%render.frag.spv
call glslc %CFLAGS% %VRTPROF% %INDIR%%RNDR%render.vert        -o %OUTDIR%%RNDR%render.vert.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%camera.comp        -o %OUTDIR%%RNDR%camera.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%clear.comp         -o %OUTDIR%%RNDR%clear.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%sampler.comp       -o %OUTDIR%%RNDR%sampler.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%filter.comp        -o %OUTDIR%%RNDR%filter.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%deinterlace.comp   -o %OUTDIR%%RNDR%deinterlace.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%rayshading.comp    -o %OUTDIR%%RNDR%rayshading.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%surface.comp       -o %OUTDIR%%RNDR%surface.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%directTraverse.comp  -o %OUTDIR%%RNDR%directTraverse.comp.spv

call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%build-new.comp     -o %OUTDIR%%HLBV%build-new.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%aabbmaker.comp     -o %OUTDIR%%HLBV%aabbmaker.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%build.comp         -o %OUTDIR%%HLBV%build.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%minmax.comp        -o %OUTDIR%%HLBV%minmax.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%refit.comp         -o %OUTDIR%%HLBV%refit.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%refit-new.comp     -o %OUTDIR%%HLBV%refit-new.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%child-link.comp    -o %OUTDIR%%HLBV%child-link.comp.spv

::call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%histogram.comp     -o %OUTDIR%%RDXI%histogram.comp.spv
::call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%permute.comp       -o %OUTDIR%%RDXI%permute.comp.spv
::call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%prefix-scan.comp   -o %OUTDIR%%RDXI%prefix-scan.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%single.comp        -o %OUTDIR%%RDXI%single.comp.spv

call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%permute.comp    -o %OUTDIR%%RDXI%permute.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%histogram.comp  -o %OUTDIR%%RDXI%histogram.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%pfx-work.comp   -o %OUTDIR%%RDXI%pfx-work.comp.spv


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

call spirv-opt %OPTFLAGS% %OUTDIR%%VRTX%loader.comp.spv         -o %OUTDIR%%VRTX%loader.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%VRTX%loader-int16.comp.spv   -o %OUTDIR%%VRTX%loader-int16.comp.spv

call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%directTraverse.comp.spv -o %OUTDIR%%RNDR%directTraverse.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%surface.comp.spv        -o %OUTDIR%%RNDR%resolver.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%rayshading.comp.spv     -o %OUTDIR%%RNDR%rayshading.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%camera.comp.spv         -o %OUTDIR%%RNDR%camera.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%sampler.comp.spv        -o %OUTDIR%%RNDR%sampler.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RNDR%clear.comp.spv          -o %OUTDIR%%RNDR%clear.comp.spv

call spirv-opt %OPTFLAGS% %OUTDIR%%HLBV%aabbmaker.comp.spv      -o %OUTDIR%%HLBV%aabbmaker.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%HLBV%build.comp.spv          -o %OUTDIR%%HLBV%build.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%HLBV%minmax.comp.spv         -o %OUTDIR%%HLBV%minmax.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%HLBV%refit.comp.spv          -o %OUTDIR%%HLBV%refit.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%HLBV%refit-new.comp.spv      -o %OUTDIR%%HLBV%refit-new.comp.spv


call spirv-opt %OPTFLAGS% %OUTDIR%%RDXI%single.comp.spv         -o %OUTDIR%%RDXI%single.comp.spv

call spirv-opt %OPTFLAGS% %OUTDIR%%RDXI%histogram.comp.spv      -o %OUTDIR%%RDXI%histogram.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RDXI%permute.comp.spv        -o %OUTDIR%%RDXI%permute.comp.spv
call spirv-opt %OPTFLAGS% %OUTDIR%%RDXI%pfx-work.comp.spv       -o %OUTDIR%%RDXI%pfx-work.comp.spv

pause
