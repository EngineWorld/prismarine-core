
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
::call glslc %CMPPROF% %INDIR%%RNDR%intersection.comp  %CFLAGS% -o %OUTDIR%%RNDR%intersection.comp.spv
::call glslc %CMPPROF% %INDIR%%EXPR%intersection.comp  %CFLAGS% -o %OUTDIR%%RNDR%intersection.comp.spv

call glslc %CMPPROF% %INDIR%%HLBV%aabbmaker.comp     %CFLAGS% -o %OUTDIR%%HLBV%aabbmaker.comp.spv
call glslc %CMPPROF% %INDIR%%HLBV%build.comp         %CFLAGS% -o %OUTDIR%%HLBV%build.comp.spv
call glslc %CMPPROF% %INDIR%%HLBV%minmax.comp        %CFLAGS% -o %OUTDIR%%HLBV%minmax.comp.spv
call glslc %CMPPROF% %INDIR%%HLBV%refit.comp         %CFLAGS% -o %OUTDIR%%HLBV%refit.comp.spv

call glslc %CMPPROF% %INDIR%%RDXI%histogram.comp     %CFLAGS% -o %OUTDIR%%RDXI%histogram.comp.spv
call glslc %CMPPROF% %INDIR%%RDXI%permute.comp       %CFLAGS% -o %OUTDIR%%RDXI%permute.comp.spv
call glslc %CMPPROF% %INDIR%%RDXI%prefix-scan.comp   %CFLAGS% -o %OUTDIR%%RDXI%prefix-scan.comp.spv


::call glslangValidator  -o %OUTDIR%%RNDR%intersection.comp.spv    %CMPPROFM% %DXDIR%%RNDR%intersection.hlsl 
::call glslangValidator  -o %OUTDIR%%HLBV%aabbmaker.comp.spv       %CMPPROFM% %DXDIR%%HLBV%aabbmaker.hlsl    
::call glslangValidator  -o %OUTDIR%%HLBV%build.comp.spv           %CMPPROFM% %DXDIR%%HLBV%build.hlsl        
::call glslangValidator  -o %OUTDIR%%HLBV%minmax.comp.spv          %CMPPROFM% %DXDIR%%HLBV%minmax.hlsl       
::call glslangValidator  -o %OUTDIR%%HLBV%refit.comp.spv           %CMPPROFM% %DXDIR%%HLBV%refit.hlsl        
::call glslangValidator  -o %OUTDIR%%RDXI%histogram.comp.spv       %CMPPROFM% %DXDIR%%RDXI%histogram.hlsl    
::call glslangValidator  -o %OUTDIR%%RDXI%permute.comp.spv         %CMPPROFM% %DXDIR%%RDXI%permute.hlsl      
::call glslangValidator  -o %OUTDIR%%RDXI%prefix-scan.comp.spv     %CMPPROFM% %DXDIR%%RDXI%prefix-scan.hlsl  

::call glslc %CMPPROF% %DXDIR%%RNDR%intersection.hlsl  %HFLAGS% -o %OUTDIR%%RNDR%intersection.comp.spv
::call glslc %CMPPROF% %DXDIR%%HLBV%aabbmaker.hlsl     %HFLAGS% -o %OUTDIR%%HLBV%aabbmaker.comp.spv
::call glslc %CMPPROF% %DXDIR%%HLBV%build.hlsl         %HFLAGS% -o %OUTDIR%%HLBV%build.comp.spv
::call glslc %CMPPROF% %DXDIR%%HLBV%minmax.hlsl        %HFLAGS% -o %OUTDIR%%HLBV%minmax.comp.spv
::call glslc %CMPPROF% %DXDIR%%HLBV%refit.hlsl         %HFLAGS% -o %OUTDIR%%HLBV%refit.comp.spv
::call glslc %CMPPROF% %DXDIR%%RDXI%histogram.hlsl     %HFLAGS% -o %OUTDIR%%RDXI%histogram.comp.spv
::call glslc %CMPPROF% %DXDIR%%RDXI%permute.hlsl       %HFLAGS% -o %OUTDIR%%RDXI%permute.comp.spv
::call glslc %CMPPROF% %DXDIR%%RDXI%prefix-scan.hlsl   %HFLAGS% -o %OUTDIR%%RDXI%prefix-scan.comp.spv

::call spirv-cross --version 450 %OUTDIR%%RNDR%intersection.comp.spv --output %OUTSHR%%RNDR%intersection.comp
::call spirv-cross --version 450 %OUTDIR%%HLBV%aabbmaker.comp.spv    --output %OUTSHR%%HLBV%aabbmaker.comp 
::call spirv-cross --version 450 %OUTDIR%%HLBV%build.comp.spv        --output %OUTSHR%%HLBV%build.comp  
::call spirv-cross --version 450 %OUTDIR%%HLBV%refit.comp.spv        --output %OUTSHR%%HLBV%refit.comp
::call spirv-cross --version 450 %OUTDIR%%RDXI%histogram.comp.spv    --output %OUTSHR%%RDXI%histogram.comp 
::call spirv-cross --version 450 %OUTDIR%%RDXI%permute.comp.spv      --output %OUTSHR%%RDXI%permute.comp  
::call spirv-cross --version 450 %OUTDIR%%RDXI%prefix-scan.comp.spv  --output %OUTSHR%%RDXI%prefix-scan.comp

pause
