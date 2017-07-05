
cd %~dp0
set CFLAGS=--target-env=opengl -x glsl -Werror 
set INDIR=.\
set OUTDIR=..\build\shaders-spv\
set VXL2=tools\
set RNDR=render\
set EXPR=experimental\
set HLBV=hlbvh\
set RDXI=radix\

set CMPPROF=-fshader-stage=compute
set FRGPROF=-fshader-stage=fragment
set VRTPROF=-fshader-stage=vertex
set GMTPROF=-fshader-stage=geometry

mkdir %OUTDIR%
mkdir %OUTDIR%%VXL2%
mkdir %OUTDIR%%RNDR%
mkdir %OUTDIR%%HLBV%
mkdir %OUTDIR%%RDXI%
mkdir %OUTDIR%%HLBV%next-gen-sort

call glslc %INDIR%%VXL2%loader.comp        %CMPPROF% %CFLAGS% -o %OUTDIR%%VXL2%loader.comp.spv -DINVERT_TX_Y
call glslc %INDIR%%VXL2%loader.comp        %CMPPROF% %CFLAGS% -o %OUTDIR%%VXL2%loader-int16.comp.spv -DINVERT_TX_Y -DENABLE_INT16_LOADING
call glslc %INDIR%%RNDR%render.frag        %FRGPROF% %CFLAGS% -o %OUTDIR%%RNDR%render.frag.spv
call glslc %INDIR%%RNDR%render.vert        %VRTPROF% %CFLAGS% -o %OUTDIR%%RNDR%render.vert.spv
call glslc %INDIR%%RNDR%begin.comp         %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%begin.comp.spv
call glslc %INDIR%%RNDR%camera.comp        %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%camera.comp.spv
call glslc %INDIR%%RNDR%clear.comp         %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%clear.comp.spv
call glslc %INDIR%%RNDR%reclaim.comp       %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%reclaim.comp.spv
call glslc %INDIR%%RNDR%sampler.comp       %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%sampler.comp.spv
call glslc %INDIR%%RNDR%testmat.comp       %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%testmat.comp.spv
call glslc %INDIR%%RNDR%testmat-rt.comp    %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%testmat-rt.comp.spv
call glslc %INDIR%%RNDR%quantizer.comp     %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%quantizer.comp.spv
call glslc %INDIR%%RNDR%intersection.comp  %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%intersection.comp.spv
call glslc %INDIR%%EXPR%intersection.comp  %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%intersection.comp.spv

call glslc %INDIR%%HLBV%aabbmaker.comp     %CMPPROF% %CFLAGS% -o %OUTDIR%%HLBV%aabbmaker.comp.spv
call glslc %INDIR%%HLBV%build.comp         %CMPPROF% %CFLAGS% -o %OUTDIR%%HLBV%build.comp.spv
call glslc %INDIR%%HLBV%minmax.comp        %CMPPROF% %CFLAGS% -o %OUTDIR%%HLBV%minmax.comp.spv
call glslc %INDIR%%HLBV%refit.comp         %CMPPROF% %CFLAGS% -o %OUTDIR%%HLBV%refit.comp.spv

call glslc %INDIR%%RDXI%histogram.comp     %CMPPROF% %CFLAGS% -o %OUTDIR%%RDXI%histogram.comp.spv
call glslc %INDIR%%RDXI%permute.comp       %CMPPROF% %CFLAGS% -o %OUTDIR%%RDXI%permute.comp.spv
call glslc %INDIR%%RDXI%prefix-scan.comp   %CMPPROF% %CFLAGS% -o %OUTDIR%%RDXI%prefix-scan.comp.spv

pause
