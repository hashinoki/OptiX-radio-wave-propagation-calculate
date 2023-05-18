// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "meidaiITC.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>

#include "sphere.h"

#define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )

extern "C" {
__constant__ Params params;
}


// ポインタをunsigned long longに変換してから、前側32bitをi0に、後側32bitをi1に格納する
static __forceinline__ __device__ void packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void* ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}


struct Payload {
  unsigned int ray_id; // unique id of the ray
  float tpath;         // total lenth of the path with multiple bounces
  float ref_idx;
  float receive;
};

static __forceinline__ __device__ void trace2(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        Payload*               prd,
        int                    offset,
        int                    stride,
        int                    miss
        )
{
    unsigned int p0, p1;
    packPointer(prd, p0, p1);
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
            p0, p1);
}static __forceinline__ __device__ Payload* getPayload2()
{
    unsigned int p0, p1;
    p0 = optixGetPayload_0();
    p1 = optixGetPayload_1();
    Payload *prd;
    prd = static_cast<Payload*>(unpackPointer(p0, p1));
    return prd;
}


static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction )
{
    float theta = static_cast<float>( idx.x ) / static_cast<float>( dim.x );
    float phi = static_cast<float>( idx.y ) / static_cast<float>( dim.y );
    float ele = M_PIf * theta;
    float azi = 2.0f * M_PIf * phi;
    //origin    = params.cam_eye;
    //origin	= make_float3(1.0f, 1.0f, 0.0f);              //for planes0
    //origin	= make_float3(-1.0f, 0.0f, 0.0f);              //for planes1  
    //origin	= make_float3(0.0f, 1.0f, 0.0f);              //for planes0	
    //origin	= make_float3(0.1f, 0.0f, 0.0f);              //for cow
    //origin    = make_float3(1.0f, 0.0f, 1.0f);              //for shpereAndPlane
    origin      = make_float3(-18161.0f, -93727.0f, 150.0f);  //for meidaiITC
    direction = make_float3(sinf(ele)*cosf(azi), sinf(ele)*sinf(azi), cosf(ele));
}


// TODO: to improve performance, pre-compute and pack the normals.
// but here we compute them while tracing
// available if BUILD_INPUT_TYPE TRIANGLES; 
__device__ __forceinline__ float3 getnormal(const unsigned int triId) {

    float3 vertex[3];
    OptixTraversableHandle gas_handle = optixGetGASTraversableHandle();
    optixGetTriangleVertexData(gas_handle, triId, 0, 0, vertex);

    float3 normal = cross((vertex[1] - vertex[0]), (vertex[2] - vertex[0]));

    return normal;
}


extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen


    float dx = static_cast<float>( idx.x ) /  static_cast<float>( dim.x );
    float dy = static_cast<float>( idx.y ) /  static_cast<float>( dim.y );
    
    //create a ray sphere for rwpl
    float3 ray_origin;
    float3 ray_direction;
    
    
    computeRay( idx, dim, ray_origin, ray_direction );
    //printf("vec(%f, %f, %f)\n", ray_direction.x, ray_direction.y, ray_direction.z);
    //printf("ray_length = %f\n", dot(ray_direction, ray_direction));
       
    // setting the per ray data (payload)
    Payload pld;
    
    pld.tpath = 0.0f;
    pld.ray_id = idx.x + dim.x * idx.y;
    pld.ref_idx = 0.0f;
    pld.receive = 0.0f;
    
    Payload *pldptr = &pld;
    //printf("%d, %d, %d, %d = %d\n", idx.x, idx.y, dim.x, dim.y, );

    
    float tmin = 1e-10f;
    float tmax = 20000.0f;

    trace2( params.handle,
        ray_origin,
        ray_direction,
        tmin,  // tmin
        tmax,  // tmax
        pldptr, 
        0,
        1,
        0               );
    
    
}


extern "C" __global__ void __miss__ms()
{
    unsigned int p0, p1;
    Payload *pldptr = getPayload2();
    packPointer(pldptr, p0, p1);
}


extern "C" __global__ void __closesthit__triangle()
{
    unsigned int tri_id = optixGetPrimitiveIndex();
    unsigned int sbt_id = optixGetSbtGASIndex();
    float time = optixGetRayTime();
    //printf("tri[%d] = sbt[%d]\n", tri_id, sbt_id);

    float3 ray_dir = optixGetWorldRayDirection();
    float3 ray_ori = optixGetWorldRayOrigin();
    //printf("dir = (%f, %f, %f)\n", ray_dir.x, ray_dir.y, ray_dir.z);
    /*
    const float3 out_normal =
    	  make_float3(
                int_as_float( optixGetAttribute_0() ),
                int_as_float( optixGetAttribute_1() ),
                int_as_float( optixGetAttribute_2() )
                );
    
    float3 vertex[3];
    OptixTraversableHandle gas_handle = optixGetGASTraversableHandle();
    optixGetTriangleVertexData(gas_handle, tri_id, sbt_id, time, vertex);
    
    float3 out_normal = cross((vertex[1] - vertex[0]), (vertex[2] - vertex[0]));
    */
    
    // printf("prim[%d] = vec(%f, %f, %f)\n", sbt_id, out_normal.x, out_normal.y, out_normal.z);
    
    // We defined out geometry as a triangle geometry. In this case the
    // We add the t value of the intersection
    float ray_tmax = optixGetRayTmax();

    Payload *pldptr = getPayload2();    
    float total_path_length = ray_tmax + pldptr->tpath;
    pldptr->tpath = total_path_length;
    
    // report individual bounces
    //printf("Ray = %d, pathlen = %f\n", pldptr->ray_id, total_path_length);


    //get vertice data from SBT and compute normal
    HitGroupData *data = (HitGroupData*)optixGetSbtDataPointer();
    const MeshData* mesh_data = (MeshData*)data->shape_data;
    const uint3 index = mesh_data->indices[tri_id];

    const float3 v0 = mesh_data->vertices[ index.x ];
    const float3 v1 = mesh_data->vertices[ index.y ];
    const float3 v2 = mesh_data->vertices[ index.z ];
    const float3 out_normal = normalize( cross((v1 - v0), (v2 - v0)));
    
    float3 hit_point = ray_ori + ray_tmax * ray_dir;
    float3 reflect_dir = reflect(ray_dir, out_normal);
    //printf("triangle vertex =(%f,%f,%f), (%f,%f,%f), (%f,%f,%f)\n", v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
    //printf("vec(%f, %f, %f)\n", reflect_dir.x, reflect_dir.y, reflect_dir.z);
    
    // cos1
    float cos1 = -1.0f * dot(ray_dir, out_normal) / (length(ray_dir) * length(out_normal));

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

    float Rv_total = pldptr->ref_idx + Rv;
    pldptr->ref_idx = Rv_total;
    
    // Minimal distance the ray has to travel to report next hit
    float tmin = 1e-5;
    float tmax = 20000.0f;    

    
    trace2(params.handle,
         hit_point,
         reflect_dir,
         tmin,  // tmin
         tmax,  // tmax
         pldptr,
         0,
         1,
         0                 );
    
    // printf("Ray = %d, pathlen = %f\n", pldptr->ray_id, pldptr->tpath);
}

extern "C" __global__ void __closesthit__sphere()
{
    Payload* pldptr = getPayload2();
    unsigned int sphe_id = optixGetPrimitiveIndex();
    
    // We defined out geometry as a triangle geometry. In this case the
    // We add the t value of the intersection
    float ray_tmax = optixGetRayTmax();

    //float4 payload = getPayload();
    float total_path_length = ray_tmax + pldptr->tpath;
    //float total_path_length = ray_tmax + payload.y;
    
    if (pldptr->ref_idx == 0.0f){
      pldptr->ref_idx = 1.0f; 
    }
    

    //float result = ((1.0 * 1.0 * 1.0) / (16 * M_PIf * M_PIf)) * (payload.z/total_path_length);
    float result = ((1.0 * 1.0 * 1.0) / (16 * M_PIf * M_PIf)) * (pldptr->ref_idx/total_path_length);
    
    pldptr->tpath = total_path_length;
    pldptr->receive = result;
    //optixSetPayload_1(__float_as_uint(total_path_length));
    //optixSetPayload_3(__float_as_uint(result));

    //float* output = params.result;
    //atomicAdd(output + sphe_id, result);
    
    //params.result[prdptr->ray_id] = result;

    //printf("Sphe[%d], result = %f\n", sphe_id, pldptr->receive);
    //printf("Sphe[%d], result = %f\n", sphe_id, params.result[sphe_id]);
    //printf("Sphe[%d], result = %f\n", sphe_id, total_path_length);
    //printf("ray[%d], result = %f\n", pldptr->ray_id, pldptr->receive);
    //printf("%d\n", pldptr->ray_id);
    //printf("%f\n", pldptr->receive);
    
    // checking for debug
    /*
    if (result < 0.005f) {
      printf("Ray = %d, pathlen = %f\n", __float_as_uint(payload.x), result);
      printf("Sphe[%d], result = %f\n", sphe_id, params.result[sphe_id]);
    }
    */
}


extern "C" __global__ void __intersection__sphere()
{
    // Shader binding tableからデータを取得
    HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    // AABBとの交差判定が認められた球体のGAS内のIDを取得
    const int prim_idx = optixGetPrimitiveIndex();
    const SphereData sphere_data = ((SphereData*)data->shape_data)[prim_idx];

    const float3 center = sphere_data.center;
    const float radius = sphere_data.radius;

    // オブジェクト空間におけるレイの原点と方向を取得
    const float3 origin = optixGetObjectRayOrigin();
    const float3 direction = optixGetObjectRayDirection();
    // レイの最小距離と最大距離を取得
    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();

    // 球体との交差判定処理（判別式を解いて、距離tを計算)
    const float3 oc = origin - center;
    const float a = dot(direction, direction);
    const float half_b = dot(oc, direction);
    const float c = dot(oc, oc) - radius * radius;

    const float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return;
    
    const float sqrtd = sqrtf(discriminant);

    float root = (-half_b - sqrtd) / a;
    if (root < tmin || tmax < root)
    {
        root = (-half_b + sqrtd) / a;
        if (root < tmin || tmax < root)
            return;
    }

    // オブジェクト空間におけるレイと球の交点を計算
    const float3 P = origin + root * direction;
    const float3 normal = (P - center) / radius;

    // 球体におけるテクスチャ座標を算出 (Z up)と仮定して、xとyから方位角、zから仰角を計算
    float phi = atan2(normal.y, normal.x);
    if (phi < 0) phi += 2.0f * M_PIf;
    const float theta = acosf(normal.z);
    const float2 texcoord = make_float2(phi / (2.0f * M_PIf), theta / M_PIf);

    // レイと球の交差判定を認める
    optixReportIntersection(root, 0, 
        __float_as_int(normal.x), __float_as_int(normal.y), __float_as_int(normal.z),
        __float_as_int(texcoord.x), __float_as_int(texcoord.y)
    );
}

