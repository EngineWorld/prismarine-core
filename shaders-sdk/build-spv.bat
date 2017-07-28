
cd %~dp0
set CFLAGS=--target-env=vulkan -x glsl -Werror 
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

set CMPPROF=-fshader-stage=comp
set FRGPROF=-fshader-stage=fragment
set VRTPROF=-fshader-stage=vertex
set GMTPROF=-fshader-stage=geometry


set CMPPROFM=-e CSMain -S comp --hlsl-iomap --target-env opengl -V -D 


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
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%testmat-rt.comp    -o %OUTDIR%%RNDR%testmat-rt.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%quantizer.comp     -o %OUTDIR%%RNDR%quantizer.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%intersection.comp  -o %OUTDIR%%RNDR%intersection.comp.spv
::call glslc %CFLAGS% %CMPPROF% %INDIR%%EXPR%intersection.comp  -o %OUTDIR%%RNDR%intersection.comp.spv

call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%aabbmaker.comp     -o %OUTDIR%%HLBV%aabbmaker.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%build.comp         -o %OUTDIR%%HLBV%build.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%minmax.comp        -o %OUTDIR%%HLBV%minmax.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%HLBV%refit.comp         -o %OUTDIR%%HLBV%refit.comp.spv

call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%histogram.comp     -o %OUTDIR%%RDXI%histogram.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%permute.comp       -o %OUTDIR%%RDXI%permute.comp.spv
call glslc %CFLAGS% %CMPPROF% %INDIR%%RDXI%prefix-scan.comp   -o %OUTDIR%%RDXI%prefix-scan.comp.spv

pause
