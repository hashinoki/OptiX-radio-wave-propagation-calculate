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
 *  * Neither the name o89 NVIDIA CORPORATION nor the names of its
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

#define PI 3.14159265359f
#include "common/vec_func.cuh"
#include "params.hpp"
#include <optix.h>

extern "C" static __constant__ Params params;


struct Payload {
  unsigned int ray_id; // unique id of the ray
  float tpath;         // total lenth of the path with multiple bounces
  float ref_idx;       // total reflection index of the path with multiple bounces
};

extern "C" __global__ void __raygen__prog() {
  const uint3 launch_index = optixGetLaunchIndex();
  const uint3 launch_dim = optixGetLaunchDimensions();

  const float3 &min_corner = params.min_corner;
  const float3 &delta = params.delta;

  //original position for mrps
  //float xo = min_corner.x + delta.x * launch_index.x;
  //float yo = min_corner.y + delta.y * launch_index.y;
  //float zo = min_corner.z;

  //direction compute for rwpl
  float dx = __uint_as_float(launch_index.x) /  __uint_as_float(launch_dim.x);
  float dy = __uint_as_float(launch_index.y) /  __uint_as_float(launch_dim.y);
  
  // setting the per ray data (payload)
  Payload pld;
  pld.tpath = 0.0f;
  pld.ray_id = launch_index.x + launch_dim.x * launch_index.y;
  pld.ref_idx = 1.0f;
  

  // create a ray for mrps
  //float3 ray_origin = make_float3(xo, yo, zo);
  //float3 ray_direction = normalize(make_float3(0.0, 0.0, 1.0));

  //create a ray for rwpl
  //float3 ray_origin = make_float3(1.0f, 0.0f, 1.0f);
  float3 ray_origin = make_float3(0.0f, 0.0f, 0.0f);
  float3 w = make_float3(-2.0, -1.0, -1.0);
  float3 u = make_float3(4.0, 0.0, 0.0);
  float3 v = make_float3(0.0, 2.0, 0.0);
  float3 ray_direction = w + dx * u + dy * v;
  //printf("dx,dy = (%f, %f)\n", dx, dy);
  //printf("dir = (%f, %f, %f)\n", ray_direction.x, ray_direction.y, ray_direction.z);
  
  float tmin = 0.0f;
  float tmax = delta.z + 100.0;
  float ray_time = 0.0f;
  OptixVisibilityMask visibilityMask = 255;
  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTIndex = 0;

  // Extract Payload as unsigned int
  unsigned int ray_id_payload = pld.ray_id;
  unsigned int tpath_payload = __float_as_uint(pld.tpath);
  unsigned int ref_idx_payload = __float_as_uint(pld.ref_idx);

  optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
             visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex,
             ray_id_payload, tpath_payload, ref_idx_payload);

  // Store back paylaod to the Payload struct
  pld.ray_id = ray_id_payload;
  pld.tpath = __uint_as_float(tpath_payload);
  pld.ref_idx = __uint_as_float(ref_idx_payload);

  // store the result and number of bounces back to the global buffers
  params.tpath[pld.ray_id] = pld.tpath;
  //printf("tpath = %f\n", params.tpath[pld.ray_id]);
}

// TODO: to improve performance, pre-compute and pack the normals.
// but here we compute them while tracing
__device__ __forceinline__ float3 outwardNormal(const unsigned int triId,
                                                const float3 rayDir) {

  float3 vertex[3];
  OptixTraversableHandle gas_handle = optixGetGASTraversableHandle();
  optixGetTriangleVertexData(gas_handle, triId, 0, 0, vertex);

  float3 normal = cross((vertex[1] - vertex[0]), (vertex[2] - vertex[0]));

  // check if the normal is facing in opposite direction of the ray dir, if yes
  // flip it in Z dir
  if (normal.z * rayDir.z < 0) {
    normal.z *= -1.0f;
  }
  return normal;
}

//if exist intersect with recieved sphere then true
__device__ float3 position(float3 ray_ori, float3 ray_dir) {
  float3 recieved_center = make_float3(0.0f, 0.0f, -5.0f);
  float  recieved_radious = 0.5f;
  float3 oc = ray_ori - recieved_center;
  float a = dot(ray_dir, ray_dir);
  float b = 2.0f * dot(oc, ray_dir);
  float c = dot(oc, oc) - (recieved_radious * recieved_radious);
  float D = b*b - 4 * a*c;
  float t1 = (-1.0f * b + sqrtf(D)) / (2.0f * a);
  float t2 = (-1.0f * b - sqrtf(D)) / (2.0f * a);
  float3 pos;
  if(D>0){
    if(t1 * t1 > t2 * t2){
      pos = ray_ori + t2 * ray_dir;
      return pos;
    }
    pos = ray_ori + t1 * ray_dir;
    return pos;
    
  }
  return recieved_center;
}

__device__ bool lastcross(float3 ray_ori, float3 ray_dir){
  float3 recieved_center = make_float3(0.0f, 0.0f, -5.0f);
  float3 pos = position(ray_ori, ray_dir);
  if(pos.x == recieved_center.x && pos.y == recieved_center.y && pos.z == recieved_center.z){
    return false;
  }
  return true;
}

__device__ bool mitooshi(float3 ray_ori, float3 ray_dir, float3 hit_point){
  float3 pos = position(ray_ori, ray_dir);
  float x = dot((hit_point - ray_ori), (hit_point - ray_ori));
  float y = dot((pos - ray_ori), (pos - ray_ori));
  if(x < y){
    return false;
  }
  return true;
}

extern "C" __global__ void __closesthit__prog() {
  unsigned int tri_id = optixGetPrimitiveIndex();
  // We defined out geometry as a triangle geometry. In this case the
  // We add the t value of the intersection
  float ray_tmax = optixGetRayTmax();

  unsigned int ray_id_payload = optixGetPayload_0();
  unsigned int tpath_payload = optixGetPayload_1();
  unsigned int ref_idx_payload = optixGetPayload_2();

  // if using t_value 
  /*
  float total_path_length = ray_tmax + __uint_as_float(tpath_payload);
  optixSetPayload_1(__float_as_uint(total_path_length));
  tpath_payload = optixGetPayload_1();
  */
  
  //printf("tpath = %f\n", ray_tmax);
 
  float3 ray_dir = optixGetWorldRayDirection();
  float3 ray_ori = optixGetWorldRayOrigin();
  //printf("dir = (%f, %f, %f)\n", ray_dir.x, ray_dir.y, ray_dir.z);


  float3 out_normal = outwardNormal(tri_id, ray_dir);
  float3 reflect_dir = reflect(ray_dir, out_normal);
  //printf("ref_dir = (%f, %f, %f)\n", reflect_dir.x, reflect_dir.y, reflect_dir.z);
  float3 hit_point = ray_ori + ray_tmax * ray_dir;
  //printf("hit_point = (%f, %f, %f)\n", hit_point.x, hit_point.y, hit_point.z);

  // if using real path length
  float total_path_length = length(hit_point - ray_ori) + __uint_as_float(tpath_payload);
  optixSetPayload_1(__float_as_uint(total_path_length));
  tpath_payload = optixGetPayload_1();

  //printf("path_length = %f\n", __uint_as_float(tpath_payload));
  
  float3 march_check = reflect_dir - ray_dir;
  float e = 1e-5;
  float total_Rv;
  if(march_check.x * march_check.x <= e && march_check.y * march_check.y <= e && march_check.z * march_check.z <= e){
     total_Rv = __uint_as_float(ref_idx_payload);
  }
  else{
     //printf("march_check = (%f, %f, %f)\n", march_check.x, march_check.y, march_check.z);
     // cos1
     float cos1 = -1.0f * dot(ray_dir, out_normal) / (length(ray_dir) * length(out_normal));
     //printf("cos1 = %f\n", cos1);

     // cos2
     float n_ij = 1.5f;
     float cos2 = sqrtf((n_ij * n_ij) - (1.0f - cos1 * cos1)) / n_ij;

     //myu
     float u1 = 1.0f;
     float u2 = 2.0f;

     //R
     //float Rp = (u1 * n_ij * cos1 - u2 * cos2)/(u1 * n_ij * n_ij * cos1 + u2 * cos2);
     float Rv = (u2 * n_ij * cos1 - u1 * cos2) / (u2 * n_ij * n_ij * cos1 + u1 * cos2);
    
     total_Rv = __uint_as_float(ref_idx_payload) * Rv;
     //printf("Rv = %f\n", Rv);
  }
  
  optixSetPayload_2(__float_as_uint(total_Rv));
  ref_idx_payload = optixGetPayload_2();

  // report individual bounces
  // printf("Ray = %d, pathlen = %f\n", payload.rayId, payload.tPath);
  // printf("Ray = %d, pathlen = %f, ref_idx = %f\n", ray_id_payload, total_path_length, total_Rv);

  float result;

  // Minimal distance the ray has to travel to report next hit
  float tmin = 1e-5;
  float tmax = params.delta.z + 100.0;
  float ray_time = 0.0f;
  OptixVisibilityMask visibilityMask = 255;
  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTIndex = 0;

  if(lastcross(ray_ori, ray_dir) == false){
    optixTrace(params.handle, hit_point, reflect_dir, tmin, tmax, ray_time,
               visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex,
               ray_id_payload, tpath_payload, ref_idx_payload);
  }
  else{
    if(mitooshi(ray_ori, ray_dir, hit_point) == false){
      optixTrace(params.handle, hit_point, reflect_dir, tmin, tmax, ray_time,
                 visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex,
                 ray_id_payload, tpath_payload, ref_idx_payload);
    }
    else{
     float total_path_length = length(position(ray_ori, ray_dir) - ray_ori) + __uint_as_float(tpath_payload);;
     
     //printf("Ray = %d, pathlen = %f, ref_idx = %f\n", ray_id_payload, total_path_length, total_Rv);
     result = ((1.0 * 1.0 * 1.0) / (16 * PI * PI)) * (total_Rv/total_path_length);
     //printf("Ray = %d, result = %f\n", ray_id_payload, result);
    }
  }
}

extern "C" __global__ void __miss__prog() {

  float3 ray_dir = optixGetWorldRayDirection();
  float3 ray_ori = optixGetWorldRayOrigin();
  //printf("ray_ori = (%f, %f, %f)\n", ray_ori.x, ray_ori.y, ray_ori.z);
  unsigned int ray_id_payload = optixGetPayload_0();
  unsigned int tpath_payload = optixGetPayload_1();
  unsigned int ref_idx_payload = optixGetPayload_2();	

  if(lastcross(ray_ori, ray_dir) == false){
    //printf("NO!\n");
  }
  else{
    //printf("path_length1 = %f\n", __uint_as_float(tpath_payload));
    //printf("path_length2 = %f\n", length(position(ray_ori, ray_dir) - ray_ori));
    //printf("hit_point = (%f, %f, %f)\n", position(ray_ori, ray_dir).x, position(ray_ori, ray_dir).y, position(ray_ori, ray_dir).z);
    //printf("ray_ori = (%f, %f, %f)\n", ray_ori.x, ray_ori.y, ray_ori.z);
    float total_path_length = length(position(ray_ori, ray_dir) - ray_ori) + __uint_as_float(tpath_payload);;
    //printf("Ray = %d, pathlen = %f, ref_idx = %f\n", ray_id_payload, total_path_length, __uint_as_float(ref_idx_payload));
    float result = ((1.0 * 1.0 * 1.0) / (16 * PI * PI)) * (__uint_as_float(ref_idx_payload)/total_path_length);
    //printf("Ray = %d, result = %f\n", ray_id_payload, result);
  }

}
// extern "C" __global__ void __anyhit__prog() {}
