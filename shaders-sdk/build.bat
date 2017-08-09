
cd %~dp0
set CFLAGS=--target-env=opengl -x glsl -Werror -Os -E -S -DUSE_OPENGL 
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

call glslc %CMPPROF% %INDIR%%VXL2%loader.comp        %CFLAGS% -o %OUTDIR%%VXL2%loader.comp -DINVERT_TX_Y
call glslc %CMPPROF% %INDIR%%VXL2%loader.comp        %CFLAGS% -o %OUTDIR%%VXL2%loader-int16.comp -DINVERT_TX_Y -DENABLE_INT16_LOADING
call glslc %FRGPROF% %INDIR%%RNDR%render.frag        %CFLAGS% -o %OUTDIR%%RNDR%render.frag
call glslc %VRTPROF% %INDIR%%RNDR%render.vert        %CFLAGS% -o %OUTDIR%%RNDR%render.vert
call glslc %CMPPROF% %INDIR%%RNDR%begin.comp         %CFLAGS% -o %OUTDIR%%RNDR%begin.comp
call glslc %CMPPROF% %INDIR%%RNDR%camera.comp        %CFLAGS% -o %OUTDIR%%RNDR%camera.comp
call glslc %CMPPROF% %INDIR%%RNDR%clear.comp         %CFLAGS% -o %OUTDIR%%RNDR%clear.comp
call glslc %CMPPROF% %INDIR%%RNDR%reclaim.comp       %CFLAGS% -o %OUTDIR%%RNDR%reclaim.comp
call glslc %CMPPROF% %INDIR%%RNDR%sampler.comp       %CFLAGS% -o %OUTDIR%%RNDR%sampler.comp
call glslc %CMPPROF% %INDIR%%RNDR%testmat.comp       %CFLAGS% -o %OUTDIR%%RNDR%testmat.comp
call glslc %CMPPROF% %INDIR%%RNDR%quantizer.comp     %CFLAGS% -o %OUTDIR%%RNDR%quantizer.comp
call glslc %CMPPROF% %INDIR%%RNDR%traverse.comp      %CFLAGS% -o %OUTDIR%%RNDR%traverse.comp
call glslc %CMPPROF% %INDIR%%RNDR%resolver.comp      %CFLAGS% -o %OUTDIR%%RNDR%resolver.comp
call glslc %CFLAGS% %CMPPROF% %INDIR%%RNDR%directTraverse.comp  -o %OUTDIR%%RNDR%directTraverse.comp

call glslc %CMPPROF% %INDIR%%HLBV%aabbmaker.comp     %CFLAGS% -o %OUTDIR%%HLBV%aabbmaker.comp
call glslc %CMPPROF% %INDIR%%HLBV%build.comp         %CFLAGS% -o %OUTDIR%%HLBV%build.comp
call glslc %CMPPROF% %INDIR%%HLBV%minmax.comp        %CFLAGS% -o %OUTDIR%%HLBV%minmax.comp
call glslc %CMPPROF% %INDIR%%HLBV%refit.comp         %CFLAGS% -o %OUTDIR%%HLBV%refit.comp

call glslc %CMPPROF% %INDIR%%RDXI%histogram.comp     %CFLAGS% -o %OUTDIR%%RDXI%histogram.comp
call glslc %CMPPROF% %INDIR%%RDXI%permute.comp       %CFLAGS% -o %OUTDIR%%RDXI%permute.comp
call glslc %CMPPROF% %INDIR%%RDXI%prefix-scan.comp   %CFLAGS% -o %OUTDIR%%RDXI%prefix-scan.comp

pause
