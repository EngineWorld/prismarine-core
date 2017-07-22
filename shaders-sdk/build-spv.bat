
cd %~dp0
set CFLAGS=--target-env=vulkan -x glsl -Werror 
set HFLAGS=--target-env=vulkan -x hlsl -Werror 
set INDIR=.\
set DXDIR=..\directx-sdk\
set OUTDIR=..\build\shaders-spv\
set VXL2=tools\
set RNDR=render\
set EXPR=experimental\
set HLBV=hlbvh\
set RDXI=radix\

set CMPPROF=-fshader-stage=comp -fentry-point=CSMain 
set FRGPROF=-fshader-stage=fragment
set VRTPROF=-fshader-stage=vertex
set GMTPROF=-fshader-stage=geometry

mkdir %OUTDIR%
mkdir %OUTDIR%%VXL2%
mkdir %OUTDIR%%RNDR%
mkdir %OUTDIR%%HLBV%
mkdir %OUTDIR%%RDXI%
mkdir %OUTDIR%%HLBV%next-gen-sort

call glslc %CMPPROF% %INDIR%%VXL2%loader.comp        %CFLAGS% -o %OUTDIR%%VXL2%loader.comp.spv -DINVERT_TX_Y
call glslc %CMPPROF% %INDIR%%VXL2%loader.comp        %CFLAGS% -o %OUTDIR%%VXL2%loader-int16.comp.spv -DINVERT_TX_Y -DENABLE_INT16_LOADING
call glslc %FRGPROF% %INDIR%%RNDR%render.frag        %CFLAGS% -o %OUTDIR%%RNDR%render.frag.spv
call glslc %VRTPROF% %INDIR%%RNDR%render.vert        %CFLAGS% -o %OUTDIR%%RNDR%render.vert.spv
call glslc %CMPPROF% %INDIR%%RNDR%begin.comp         %CFLAGS% -o %OUTDIR%%RNDR%begin.comp.spv
call glslc %CMPPROF% %INDIR%%RNDR%camera.comp        %CFLAGS% -o %OUTDIR%%RNDR%camera.comp.spv
call glslc %CMPPROF% %INDIR%%RNDR%clear.comp         %CFLAGS% -o %OUTDIR%%RNDR%clear.comp.spv
call glslc %CMPPROF% %INDIR%%RNDR%reclaim.comp       %CFLAGS% -o %OUTDIR%%RNDR%reclaim.comp.spv
call glslc %CMPPROF% %INDIR%%RNDR%sampler.comp       %CFLAGS% -o %OUTDIR%%RNDR%sampler.comp.spv
call glslc %CMPPROF% %INDIR%%RNDR%testmat.comp       %CFLAGS% -o %OUTDIR%%RNDR%testmat.comp.spv
call glslc %CMPPROF% %INDIR%%RNDR%testmat-rt.comp    %CFLAGS% -o %OUTDIR%%RNDR%testmat-rt.comp.spv
call glslc %CMPPROF% %INDIR%%RNDR%quantizer.comp     %CFLAGS% -o %OUTDIR%%RNDR%quantizer.comp.spv
call glslc %CMPPROF% %INDIR%%RNDR%intersection.comp  %CFLAGS% -o %OUTDIR%%RNDR%intersection.comp.spv
::call glslc %CMPPROF% %DXDIR%%RNDR%intersection.hlsl  %HFLAGS% -o %OUTDIR%%RNDR%intersection.comp.spv
::call glslc %CMPPROF% %INDIR%%EXPR%intersection.comp  %CFLAGS% -o %OUTDIR%%RNDR%intersection.comp.spv

call glslc %CMPPROF% %INDIR%%HLBV%aabbmaker.comp     %CFLAGS% -o %OUTDIR%%HLBV%aabbmaker.comp.spv
call glslc %CMPPROF% %INDIR%%HLBV%build.comp         %CFLAGS% -o %OUTDIR%%HLBV%build.comp.spv
call glslc %CMPPROF% %INDIR%%HLBV%minmax.comp        %CFLAGS% -o %OUTDIR%%HLBV%minmax.comp.spv
call glslc %CMPPROF% %INDIR%%HLBV%refit.comp         %CFLAGS% -o %OUTDIR%%HLBV%refit.comp.spv

call glslc %CMPPROF% %INDIR%%RDXI%histogram.comp     %CFLAGS% -o %OUTDIR%%RDXI%histogram.comp.spv
call glslc %CMPPROF% %INDIR%%RDXI%permute.comp       %CFLAGS% -o %OUTDIR%%RDXI%permute.comp.spv
call glslc %CMPPROF% %INDIR%%RDXI%prefix-scan.comp   %CFLAGS% -o %OUTDIR%%RDXI%prefix-scan.comp.spv

pause
