
cd %~dp0
set CFLAGS=--target-env=opengl -x glsl -Werror -Os -E -S
set INDIR=.\
set OUTDIR=..\build\shaders\
set VXL2=tools\
set RNDR=render\
set HLBV=hlbvh\
set RDXI=radix\
set EXPR=experimental\

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

call glslc %INDIR%%VXL2%loader.comp        %CMPPROF% %CFLAGS% -o %OUTDIR%%VXL2%loader.comp -DINVERT_TX_Y
call glslc %INDIR%%VXL2%loader.comp        %CMPPROF% %CFLAGS% -o %OUTDIR%%VXL2%loader-int16.comp -DINVERT_TX_Y -DENABLE_INT16_LOADING
call glslc %INDIR%%RNDR%render.frag        %FRGPROF% %CFLAGS% -o %OUTDIR%%RNDR%render.frag
call glslc %INDIR%%RNDR%render.vert        %VRTPROF% %CFLAGS% -o %OUTDIR%%RNDR%render.vert
call glslc %INDIR%%RNDR%begin.comp         %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%begin.comp
call glslc %INDIR%%RNDR%camera.comp        %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%camera.comp
call glslc %INDIR%%RNDR%clear.comp         %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%clear.comp
call glslc %INDIR%%RNDR%reclaim.comp       %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%reclaim.comp
call glslc %INDIR%%RNDR%sampler.comp       %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%sampler.comp
call glslc %INDIR%%RNDR%testmat.comp       %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%testmat.comp
call glslc %INDIR%%RNDR%testmat-rt.comp    %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%testmat-rt.comp
call glslc %INDIR%%RNDR%quantizer.comp     %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%quantizer.comp
call glslc %INDIR%%RNDR%intersection.comp  %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%intersection.comp
::call glslc %INDIR%%EXPR%intersection.comp  %CMPPROF% %CFLAGS% -o %OUTDIR%%RNDR%intersection.comp

call glslc %INDIR%%HLBV%aabbmaker.comp     %CMPPROF% %CFLAGS% -o %OUTDIR%%HLBV%aabbmaker.comp
call glslc %INDIR%%HLBV%build.comp         %CMPPROF% %CFLAGS% -o %OUTDIR%%HLBV%build.comp
call glslc %INDIR%%HLBV%minmax.comp        %CMPPROF% %CFLAGS% -o %OUTDIR%%HLBV%minmax.comp
call glslc %INDIR%%HLBV%refit.comp         %CMPPROF% %CFLAGS% -o %OUTDIR%%HLBV%refit.comp
call glslc %INDIR%%HLBV%resort.comp        %CMPPROF% %CFLAGS% -o %OUTDIR%%HLBV%resort.comp

call glslc %INDIR%%RDXI%histogram.comp     %CMPPROF% %CFLAGS% -o %OUTDIR%%RDXI%histogram.comp
call glslc %INDIR%%RDXI%permute.comp       %CMPPROF% %CFLAGS% -o %OUTDIR%%RDXI%permute.comp
call glslc %INDIR%%RDXI%prefix-scan.comp   %CMPPROF% %CFLAGS% -o %OUTDIR%%RDXI%prefix-scan.comp

pause
