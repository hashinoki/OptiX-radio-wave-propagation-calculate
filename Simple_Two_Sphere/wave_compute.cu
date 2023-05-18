//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "wave_compute.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>

# define MY_PI 3.14159

extern "C" {
__constant__ Params params;
}


static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float4*                prd,
        int                    offset,
        int                    stride,
        int                    miss
        )
{
    unsigned int p0, p1, p2, p3;
    p0 = float_as_int( prd->x );
    p1 = float_as_int( prd->y );
    p2 = float_as_int( prd->z );
    p3 = float_as_int( prd->w );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            offset,                   // SBT offset
            stride,                   // SBT stride(obj_count - 1)
            miss,                     // missSBTIndex
            p0, p1, p2, p3);
    prd->x = int_as_float( p0 );
    prd->y = int_as_float( p1 );
    prd->z = int_as_float( p2 );
    prd->w = int_as_float( p3 );
}



static __forceinline__ __device__ void setPayload( float4 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
    optixSetPayload_3( float_as_int( p.w ) );
}


static __forceinline__ __device__ float4 getPayload()
{
    return make_float4(
            int_as_float( optixGetPayload_0() ),
            int_as_float( optixGetPayload_1() ),
            int_as_float( optixGetPayload_2() ),
            int_as_float( optixGetPayload_3() )
            );
}



struct Payload {
  unsigned int ray_id; // unique id of the ray
  float tpath;         // total lenth of the path with multiple bounces
  float ref_idx;
  float recieve;
};


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float3 &delta = params.delta;

    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    const float3      U      = rtData->camera_u;
    const float3      V      = rtData->camera_v;
    const float3      W      = rtData->camera_w;
    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    const float3 origin      = rtData->cam_eye;
    const float3 direction   = normalize( d.x * U + d.y * V + W );

    
    // setting the per ray data (payload)
    Payload pld;
    pld.tpath = 0.0f;
    pld.ray_id = idx.x + dim.x * idx.y;
    pld.ref_idx = 0.0f;
    pld.recieve = 0.0f;

    // Extract Payload as unsigned int
    float ray_id = __uint_as_float(pld.ray_id);

    float tpath = pld.tpath;
    float ref_idx = pld.ref_idx;
    float result = pld.recieve;

    float4 payload_id_length = make_float4(ray_id, tpath, ref_idx, result);
    
    float tmin = 0.0f;
    float tmax = delta.z + 100.0;
    
    trace( params.handle,
        origin,
        direction,
        tmin,  // tmin
        tmax,  // tmax
        &payload_id_length, 
        0,
        1,
        0               );
    

    // for rendering store the result of color compute back to the global buffer
    //params.image[idx.y * params.image_width + idx.x] = make_color( payload_rgb );

    // Store back paylaod to the Payload struct
     pld.ray_id = __float_as_uint(payload_id_length.x);
     pld.tpath = payload_id_length.y;
     pld.ref_idx = payload_id_length.z;
     pld.recieve = payload_id_length.w;

    // store the result and  number of bounces back to the global buffers
    params.tpath[pld.ray_id] = pld.tpath;
    params.result[pld.ray_id] = pld.recieve;

    // report result of bounds
    /*
    if (pld.tpath > 0.0f) {
        printf("result of ray[%d]path is :%f\n", pld.ray_id, pld.tpath);
    }
    
    if (pld.recieve > 0.0f){
        printf("ray[%d] wave = %f\n", pld.ray_id, pld.recieve);
    }
    */
}

extern "C" __global__ void __miss__ms()
{
    //MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float4 payload = getPayload();
    setPayload( payload );
}

/*
// TODO: to improve performance, pre-compute and pack the normals.
// but here we compute them while tracing
// available if BUILD_INPUT_TYPE TRIANGLES; 
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
*/


extern "C" __global__ void __closesthit__ch__metal()
{
    /*
     // color compute and set payload
    const float3 shading_normal =
        make_float3(
                int_as_float( optixGetAttribute_0() ),
                int_as_float( optixGetAttribute_1() ),
                int_as_float( optixGetAttribute_2() )
                );

    float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( shading_normal ) );
    float3 out_normal = faceforward(world_normal, -optixGetWorldRayDirection(), world_normal);
    //printf("vec(%f, %f, %f)\n", ffnormal.x, ffnormal.y, ffnormal.z);
    */

    
    //unsigned int tri_id = optixGetPrimitiveIndex();
    //printf("tri_id = %d\n", tri_id);

    // We defined out geometry as a triangle geometry. In this case the
    // We add the t value of the intersection
    float ray_tmax = optixGetRayTmax();

    float4 payload = getPayload();
    
    float total_path_length = ray_tmax + payload.y;
    optixSetPayload_1(__float_as_uint(total_path_length));
  
    // report individual bounces
    //printf("Ray = %d, pathlen = %f\n", __float_as_uint(payload.x), total_path_length);
  
    float3 ray_dir = optixGetWorldRayDirection();
    float3 ray_ori = optixGetWorldRayOrigin();
    
    float3 hit_point = ray_ori + ray_tmax * ray_dir;
    //printf("t value = %f\n", ray_tmax);
    //printf("vec(%f, %f, %f)\n", hit_point.x, hit_point.y, hit_point.z);

    
    float3 center = { -0.6f, 0.0f, -1.0f };
    float radius = 0.5f;
    float3 out_normal = (hit_point - center) / radius;
    //printf("vec(%f, %f, %f)\n", out_normal.x, out_normal.y, out_normal.z);
    

    float3 reflect_dir = reflect(ray_dir, out_normal);
    //printf("vec(%f, %f, %f)\n", reflect_dir.x, reflect_dir.y, reflect_dir.z);
    
    
    // cos1
    float cos1 = -1.0f * dot(ray_dir, out_normal) / (length(ray_dir) * length(out_normal));
    //printf("cos1 = %f\n", cos1);

    //cos2
    float n_ij = 1.5f;
    float cos2 = sqrtf((n_ij * n_ij) - (1.0f - cos1 * cos1)) / n_ij;
    //printf("cos2 = %f\n", cos2);
    
    //myu
    float u1 = 1.0f;
    float u2 = 2.0f;

    //R
    //float Rp = (u1 * n_ij * cos1 - u2 * cos2)/(u1 * n_ij * n_ij * cos1 + u2 * cos2);
    float Rv = (u2 * n_ij * cos1 - u1 * cos2) / (u2 * n_ij * n_ij * cos1 + u1 * cos2);
    
    float total_Rv = payload.z + Rv;
    //printf("total_Rv = %f\n", Rv);
    optixSetPayload_2(__float_as_uint(total_Rv));

    float4 payload_id_length = getPayload();
    

    // Minimal distance the ray has to travel to report next hit
    float tmin = 1e-5;
    float tmax = params.delta.z + 100.0;    

    trace(params.handle,
         hit_point,
         reflect_dir,
         tmin,  // tmin
         tmax,  // tmax
         &payload_id_length,
         1,
         1,
         0                 );
    

    //printf("Ray = %d, pathlen = %f\n", __float_as_uint(payload_id_length.x), payload.y);
    
}


extern "C" __global__ void __closesthit__ch__normal()
{
    // We defined out geometry as a triangle geometry. In this case the
    // We add the t value of the intersection
    float ray_tmax = optixGetRayTmax();

    float4 payload = getPayload();

    //if (payload.y > 0.0f) {
    printf("Ray = %d, pathlen = %f\n", __float_as_uint(payload.x), payload.y);

    float total_path_length = ray_tmax + payload.y;

    if (payload.z == 0.0f) {
        payload.z = 1.0f;
    }

    float result = ((1.0 * 1.0 * 1.0) / (16 * MY_PI * MY_PI)) * (payload.z/total_path_length);

    optixSetPayload_1(__float_as_uint(total_path_length));
    optixSetPayload_3(__float_as_uint(result));

    // checking for debug 
    //if (result < 0.005f) {
    //printf("Ray = %d, pathlen = %f\n", __float_as_uint(payload.x), result);
}

