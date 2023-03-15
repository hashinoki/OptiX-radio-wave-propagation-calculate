/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include "common/common.h"
#include "common/Timer.h"

#include "maxBounce.h"
#include "params.hpp"
#include "rtxFunctions.hpp"


RTXDataHolder *rtx_dataholder;

// small problem for debugging
//uint32_t width = 8u;  // buffer size x
//uint32_t height = 8u; // buffer size y
uint32_t depth = 8u;

// fits on a single screen
// uint32_t height = 32u; // buffer size y
// uint32_t width = 32u;  // buffer size x

// test
uint32_t  height = 20000u;   // buffer size y
uint32_t  width  = 40000u;   // buffer size x

// 100M rays
// uint32_t  height = 10000u;   // buffer size y
// uint32_t  width  = 20000u;   // buffer size x

int main(int argc, char **argv) {
  std::string testfile = "sphereAndPlane";
  std::string obj_file = OBJ_DIR "./" + testfile + ".obj";
  std::cout << "Mesh file = " << obj_file << std::endl;
  std::string ptx_filename = BUILD_DIR "/ptx/optixPrograms.ptx";

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  
  cudaStreamSynchronize(stream);
  Timer timer1;
	
  ///////////////////////////////////////////////////////////////////////////////
	//example of description
  ///////////////////////////////////////////////////////////////////////////////
  rtx_dataholder = new RTXDataHolder();
  std::cout << "Initializing Context \n";
  rtx_dataholder->initContext();
  std::cout << "Reading PTX file and creating modules \n";
  rtx_dataholder->createModule(ptx_filename);
  std::cout << "Creating Optix Program Groups \n";
  rtx_dataholder->createProgramGroups();
  std::cout << "Linking Pipeline \n";
  rtx_dataholder->linkPipeline();
  std::cout << "Building Shader Binding Table (SBT) \n";
  rtx_dataholder->buildSBT();
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  cudaStreamSynchronize(stream);
  timer1.stop();
  cudaStreamSynchronize(stream);
  Timer timer2;
  
  std::vector<float3> vertices;
  std::vector<uint3> triangles;
  std::cout << "Building Acceleration Structure \n";
  OptixAabb aabb_box =
      rtx_dataholder->buildAccelerationStructure(obj_file, vertices, triangles);

  // calculate delta
  float3 delta = make_float3((aabb_box.maxX - aabb_box.minX) / width,
                             (aabb_box.maxY - aabb_box.minY) / height,
                             (aabb_box.maxZ - aabb_box.minZ) / depth);
  float3 min_corner = make_float3(aabb_box.minX, aabb_box.minY, aabb_box.minZ);

  //initialization of array for params.tpath
  /*
  float *array;
  array = (float *)malloc(sizeof(float)*width*height);
  for(int i = 0; i<width*height; i++){
    array[i] = 0.0f;
  }

  for(int i = 0; i<width*height; i++){
    std::cout << array[i] << std::endl;
  }
  */

  // check how much size memory can use
  //size_t size;
  //cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
  //printf("Heap Size=%ld\n", size);
  //printf("float size = %ld\n", sizeof(float));
  //printf("unsigned int size = %ld\n", sizeof(unsigned int));
  //printf("float3 size = %ld\n", sizeof(float3));
  //printf("OptiXTraversableHandle size = %ld\n", sizeof(OptixTraversableHandle));
  
  float *d_tpath;
  //d_tpath = array;  
  CUDA_CHECK(cudaMalloc((void **)&d_tpath, width * height * sizeof(float)));
  printf("array size = %ld\n", sizeof(float) * width * height);
  
  /*
  for(int i = 0; i<width*height; i++){
    std::cout << &d_tpath[i] << std::endl;
  }
  */  
  //printf("Params size = %ld\n", sizeof(Params));
  
  // Algorithmic parameters and data pointers used in GPU program
  Params params;
  params.min_corner = min_corner;
  params.delta = delta;
  params.handle = rtx_dataholder->gas_handle;
  params.width = width;
  params.height = height;
  params.depth = depth;
  //params.tpath = (float*)malloc(width * height * sizeof(float));
  params.tpath = d_tpath;

  Params *d_param;
  CUDA_CHECK(cudaMalloc((void **)&d_param, sizeof(Params)));
  CUDA_CHECK(
      cudaMemcpy(d_param, &params, sizeof(params), cudaMemcpyHostToDevice));

  const OptixShaderBindingTable &sbt = rtx_dataholder->sbt;
  cudaStreamSynchronize(stream);
  timer2.stop();

  //We synchronize here for the timer
  cudaStreamSynchronize(stream);
  Timer timer3;
  
  //std::cout << "Launching Ray Tracer to scatter rays \n";
  OPTIX_CHECK(optixLaunch(rtx_dataholder->pipeline, stream,
                          reinterpret_cast<CUdeviceptr>(d_param),
                          sizeof(Params), &sbt, width, height, 1));

  cudaStreamSynchronize(stream);
  timer3.stop();
  
  CUDA_CHECK(
      cudaMemcpy(&params, d_param, sizeof(params), cudaMemcpyDeviceToHost));
  
  CUDA_CHECK(cudaDeviceSynchronize());

  /*
  for(int i = 0; i<width*height; i++){
    std::cout << params.tpath[i] << std::endl;
  }
  */
  
  std::cout << std::setw(32) << std::left << "optix setting took: " << std::setw(12)
	    << std::right << timer1.get_elapsed_s() - timer3.get_elapsed_s()<< "s\n";
  std::cout << std::setw(32) << std::left << "load obj and set param took: " << std::setw(12)
	    << std::right << timer2.get_elapsed_s() - timer3.get_elapsed_s()<< "s\n";
  std::cout << std::setw(32) << std::left << "Launch took: " << std::setw(12)
	    << std::right << timer3.get_elapsed_s() << "s\n";
  std::cout << "Cleaning up ... \n";
  CUDA_CHECK(cudaFree(d_tpath));
  CUDA_CHECK(cudaFree(d_param));
  delete rtx_dataholder;

  return 0;
}
