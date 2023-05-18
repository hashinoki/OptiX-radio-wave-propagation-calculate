//
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
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sutil/Timer.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <sutil/tiny_obj_loader.h>

#include "meidaiITC.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>



template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

const uint32_t OBJ_COUNT = 2;


void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
{
    cam.setEye( {0.0f, 2.0f, 0.0f} );
    //cam.setLookat( {0.0f, 0.0f, 0.0f} );
    //cam.setUp( {0.0f, 1.0f, 3.0f} );
    //cam.setFovY( 45.0f );
    //cam.setAspectRatio( (float)width / (float)height );
}


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit( 1 );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

// Geometry acceleration structure (GAS) 用
// GASのtraversable handleをOptixInstanceに紐づける際に、
// GASが保持するSBT recordの数がわかると、
// Instanceのsbt offsetを一括で構築しやすい
struct GeometryAccelData
{
    OptixTraversableHandle handle;
    CUdeviceptr d_output_buffer;
    uint32_t num_sbt_records;
};

// Instance acceleration structure (IAS) 用
// 
struct InstanceAccelData
{
    OptixTraversableHandle handle;
    CUdeviceptr d_output_buffer;

    // IASを構築しているOptixInstanceのデータを更新できるように、
    // デバイス側のポインタを格納しておく
    CUdeviceptr d_instances_buffer;
};

// -----------------------------------------------------------------------
// Geometry acceleration structureの構築
// -----------------------------------------------------------------------
void buildGAS(OptixDeviceContext context, GeometryAccelData& gas, OptixBuildInput& build_input)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; //compactionを許可
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;        //AS更新の際はOPERATION_UPDATE

    //ASのビルドに必要なメモリ領域を計算
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        &build_input,
        1, // Number of build inputs
        &gas_buffer_sizes
    ));

    //ASを構築するための一時バッファを確保
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes) );

    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compacted_size_offset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
        compacted_size_offset + 8
    ));

    // 新たな出力バッファを確保する必要がある場合には、OptixAccelEmitDesc::result を
    // デバイス(GPU)側からホスト(CPU)側へコピーする必要がある。
    OptixAccelEmitDesc emit_property = {};
    emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compacted_size_offset);

    //ASのビルド
    OPTIX_CHECK(optixAccelBuild(
        context,
        0,                  // CUDA stream
        &accel_options,
        &build_input,
        1,                  // num build inputs
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &gas.handle,
        &emit_property,            // emitted property list
        1                   // num emitted properties
    ));

    //一時バッファは必要ないので開放
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emit_property.result, sizeof(size_t), cudaMemcpyDeviceToHost));
    //compaction後の領域が、compaction前の領域のサイズより小さい場合のみCompactionを行う
    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas.d_output_buffer), compacted_gas_size));
        OPTIX_CHECK(optixAccelCompact(context, 0, gas.handle, gas.d_output_buffer, compacted_gas_size, &gas.handle));
        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        gas.d_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

OptixAabb sphereBound(const SphereData& sphere)
{
    //球体のAxis-aligned bounding box(AABB)を返す
    const float3 center = sphere.center;
    const float radius = sphere.radius;
    return OptixAabb{
        /* minX = */ center.x - radius, /* minY = */ center.y - radius, /* minZ = */ center.z - radius,
        /* maxX = */ center.x + radius, /* maxY = */ center.y + radius, /* maxZ = */ center.z + radius
    };
}

uint32_t getNumSbtRecords(const std::vector<uint32_t>& sbt_indices)
{
    std::vector<uint32_t> sbt_counter;
    for (const uint32_t& sbt_idx : sbt_indices)
    {
        auto itr = std::find(sbt_counter.begin(), sbt_counter.end(), sbt_idx);
        if (sbt_counter.empty() || itr == sbt_counter.end())
            sbt_counter.emplace_back(sbt_idx);
    }
    return static_cast<uint32_t>(sbt_counter.size());
}

void* buildSphereGAS(OptixDeviceContext context,
    GeometryAccelData& gas,
    const std::vector<SphereData>& spheres,
    const std::vector<uint32_t>& sbt_indices
)
{
    //sphere配列からAABBの配列を作る
    std::vector<OptixAabb> aabb;
    std::transform(spheres.begin(), spheres.end(), std::back_inserter(aabb),
        [](const SphereData& sphere) { return sphereBound(sphere); });

    //AABBの配列をGPU上にコピー
    CUdeviceptr d_aabb_buffer;
    const size_t aabb_size = sizeof(OptixAabb) * aabb.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), aabb_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_aabb_buffer),
        aabb.data(), aabb_size,
        cudaMemcpyHostToDevice
    ));

    //Instance sbt offsetを基準としたsbt indexの配列をGPU上にコピー
    CUdeviceptr d_sbt_indices;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_indices), sizeof(uint32_t) * sbt_indices.size()));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_sbt_indices),
        sbt_indices.data(), sizeof(uint32_t) * sbt_indices.size(),
        cudaMemcpyHostToDevice
    ));

    // 全球体データの配列をGPU上にコピー
    // 個々の球体データへのアクセスはoptixGetPrimitiveIndex()を介して行う
    void* d_sphere;
    CUDA_CHECK(cudaMalloc(&d_sphere, sizeof(SphereData) * spheres.size()));
    CUDA_CHECK(cudaMemcpy(d_sphere, spheres.data(), sizeof(SphereData) * spheres.size(), cudaMemcpyHostToDevice));

    //重複のないsbt_indexの個数を数える
    uint32_t num_sbt_records = getNumSbtRecords(sbt_indices);
    gas.num_sbt_records = num_sbt_records;

    // 重複のないsbt_indexの分だけflagsを設定する
    // Anyhit プログラムを使用したい場合はFLAG_NONE or FLAG_REQUIRE_SINGLE_ANYHIT_CALL に設定する
    uint32_t* input_flags = new uint32_t[num_sbt_records];
    for (uint32_t i = 0; i < num_sbt_records; i++)
        input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    // Custom primitives用のAABB配列やSBTレコードのインデックス配列を
    // build input に設定する
    // num_sbt_recordsはあくまでSBTレコードの数でプリミティブ数でないことに注意
    OptixBuildInput sphere_input = {};
    sphere_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    sphere_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
    sphere_input.customPrimitiveArray.numPrimitives = static_cast<uint32_t>(spheres.size());
    sphere_input.customPrimitiveArray.flags = input_flags;
    sphere_input.customPrimitiveArray.numSbtRecords = num_sbt_records;
    sphere_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_indices;
    sphere_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    sphere_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    buildGAS(context, gas, sphere_input);
 
    return d_sphere;
}

void* buildTriangleGAS(OptixDeviceContext context,
    GeometryAccelData& gas,
    const std::vector<float3>& vertices,
    std::vector<uint3>& triangles,
    const std::vector<uint32_t>& sbt_indices
)
{
    //Instance sbt offsetを基準としたsbt indexの配列をGPU上にコピー
    CUdeviceptr d_sbt_indices;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_indices), sizeof(uint32_t) * sbt_indices.size()));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_sbt_indices),
        sbt_indices.data(), sizeof(uint32_t) * sbt_indices.size(),
        cudaMemcpyHostToDevice
    ));

    //全三角形頂点データの配列をGPU上にコピー
    const size_t vertices_size = sizeof(float3) * vertices.size();
    CUdeviceptr d_vertices = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vertices),
        vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice
    ));

    //全三角形のつなぎ方を示すインデックスデータ配列をGPU上にコピー
    const size_t tri_size = sizeof(uint3) * triangles.size();
    CUdeviceptr d_triangles = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangles), tri_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_triangles),
        triangles.data(), 
        tri_size,
        cudaMemcpyHostToDevice)
    );

    // メッシュデータを構造体に格納し、GPU上にコピー
    void* d_mesh_data;
    MeshData mesh_data{ reinterpret_cast<float3*>(d_vertices), reinterpret_cast<uint3*>(d_triangles) };
    CUDA_CHECK(cudaMalloc(&d_mesh_data, sizeof(MeshData)));
    CUDA_CHECK(cudaMemcpy(
        d_mesh_data, &mesh_data, sizeof(MeshData), cudaMemcpyHostToDevice
    ));

    //重複のないsbt_indexの個数を数える
    uint32_t num_sbt_records = getNumSbtRecords(sbt_indices);
    gas.num_sbt_records = num_sbt_records;

    // 重複のないsbt_indexの分だけflagsを設定する
    // Anyhit プログラムを使用したい場合はFLAG_NONE or FLAG_REQUIRE_SINGLE_ANYHIT_CALL に設定する
    uint32_t* input_flags = new uint32_t[num_sbt_records];
    for (uint32_t i = 0; i < num_sbt_records; i++)
        input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    // メッシュの頂点情報やインデックスバッファ、SBTレコードのインデックス配列をbuild inputに設定
    // num_sbt_recordsはあくまでSBTレコードの数で三角形の数でないことに注意
    OptixBuildInput mesh_input = {};
    mesh_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    mesh_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    mesh_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    mesh_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    mesh_input.triangleArray.vertexBuffers = &d_vertices;
    mesh_input.triangleArray.flags = input_flags;
    mesh_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    mesh_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    mesh_input.triangleArray.indexBuffer = d_triangles;
    mesh_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(triangles.size());
    mesh_input.triangleArray.numSbtRecords = num_sbt_records;
    mesh_input.triangleArray.sbtIndexOffsetBuffer = d_sbt_indices;
    mesh_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    mesh_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    buildGAS(context, gas, mesh_input);

    return d_mesh_data;

}

/*
void buildIAS(OptixDeviceContext context, 
    InstanceAccelData ias, 
    const std::vector<OptixInstance> &instances
)
*/


// -----------------------------------------------------------------------
// Instance acceleration structureの構築
// -----------------------------------------------------------------------
void buildIAS(OptixDeviceContext context, InstanceAccelData& ias, const std::vector<OptixInstance>& instances)
{
    CUdeviceptr d_instances;
    const size_t instances_size = sizeof(OptixInstance) * instances.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instances_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_instances),
        instances.data(), instances_size,
        cudaMemcpyHostToDevice
    ));

    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances = d_instances;
    instance_input.instanceArray.numInstances = static_cast<uint32_t>(instances.size());

    OptixAccelBuildOptions accel_options = {};
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    
    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        &instance_input,
        1, // num build input
        &ias_buffer_sizes
    ));

    size_t d_temp_buffer_size = ias_buffer_sizes.tempSizeInBytes;

    // ASを構築するための一時バッファを確保
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer),
        d_temp_buffer_size
    ));

    CUdeviceptr d_buffer_temp_output_ias_and_compacted_size;
    size_t compacted_size_offset = roundUp<size_t>(ias_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_ias_and_compacted_size),
        compacted_size_offset + 8
    ));

    // Compaction後のデータ領域を確保するためのEmit property
    OptixAccelEmitDesc emit_property = {};
    emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result = (CUdeviceptr)((char*)d_buffer_temp_output_ias_and_compacted_size + compacted_size_offset);

    // ASのビルド
    OPTIX_CHECK(optixAccelBuild(
        context,
        0,
        &accel_options,
        &instance_input,
        1,                  // num build inputs
        d_temp_buffer,
        d_temp_buffer_size,
        // ias.d_output_buffer,
        d_buffer_temp_output_ias_and_compacted_size,
        ias_buffer_sizes.outputSizeInBytes,
        &ias.handle,        // emitted property list
        nullptr,            // num emitted property
        0
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    
    size_t compacted_ias_size;
    CUDA_CHECK(cudaMemcpy(&compacted_ias_size, (void*)emit_property.result, sizeof(size_t), cudaMemcpyDeviceToHost));
    
    if (compacted_ias_size < ias_buffer_sizes.outputSizeInBytes && compacted_ias_size > 0)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ias.d_output_buffer), compacted_ias_size));
        OPTIX_CHECK(optixAccelCompact(context, 0, ias.handle, ias.d_output_buffer, compacted_ias_size, &ias.handle));
        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_ias_and_compacted_size));
    }
    else
    {
        ias.d_output_buffer = d_buffer_temp_output_ias_and_compacted_size;
    }
    
    // if error compaction
    //ias.d_output_buffer = d_buffer_temp_output_ias_and_compacted_size;
}

OptixAabb read_obj_mesh(const std::string &obj_filename,
                        std::vector<float3> &vertices,
                        std::vector<uint3> &triangles) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string warn;
  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                              obj_filename.c_str());
  OptixAabb aabb;
  aabb.minX = aabb.minY = aabb.minZ = std::numeric_limits<float>::max();
  aabb.maxX = aabb.maxY = aabb.maxZ = -std::numeric_limits<float>::max();

  if (!err.empty()) {
    std::cerr << err << std::endl;
    return aabb;
  }

  for (size_t s = 0; s < shapes.size(); s++) {
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      int fv = shapes[s].mesh.num_face_vertices[f];

      auto vertexOffset = vertices.size();

      for (size_t v = 0; v < fv; v++) {
        // access to vertex
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

        if (idx.vertex_index >= 0) {
          tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
          tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
          tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

          vertices.push_back(make_float3(vx, vy, vz));

          // Update aabb
          aabb.minX = std::min(aabb.minX, vx);
          aabb.minY = std::min(aabb.minY, vy);
          aabb.minZ = std::min(aabb.minZ, vz);

          aabb.maxX = std::max(aabb.maxX, vx);
          aabb.maxY = std::max(aabb.maxY, vy);
          aabb.maxZ = std::max(aabb.maxZ, vz);
        }
      }
      index_offset += fv;

      triangles.push_back(
          make_uint3(vertexOffset, vertexOffset + 1, vertexOffset + 2));
    }
  }
  return aabb;
}

enum class ShapeType
{
    Mesh,
    Sphere
};

int main( int argc, char* argv[] )
{
    std::string outfile;
    int         width  = 10000; //1024
    int         height = 20000;  //768
    int         num_sphere  = 1; //101
    
    //std::string testfile = "sphereAndPlane";
    
    std::string testfile = "meidai_ITC";
    std::string obj_file = "../resources/" + testfile + ".obj";
    
    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i < argc - 1 )
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        char log[2048]; // For error reporting from OptiX creation functions


        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK( cudaFree( 0 ) );

            // Initialize the OptiX API, loading all API entry points
            OPTIX_CHECK( optixInit() );

            // Specify context options
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction       = &context_log_cb;
            options.logCallbackLevel          = 4;

            // Associate a CUDA context (and therefore a specific GPU) with this
            // device context
            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
        }

        //もしジオメトリごとにmaterialが複数ある場合はこれを利用
        //std::vector<std::pair<char, void*>> hitgroup_datas;

        //
        // accel handling
        //
        OptixTraversableHandle gas_handle;
        InstanceAccelData ias;
        CUdeviceptr d_gas_output_buffer;
        std::vector<std::pair<ShapeType, HitGroupData>> hitgroup_datas;
        {
            // Use default options for simplicity.  In a real use case we would want to
            // enable compaction, etc
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	    /*
            // Triangle build input: simple list of three vertices
            const std::vector<float3> vertices =
            {
                { -0.5f,  0.0f, -0.5f },
                {  0.5f,  0.0f,  0.5f },
                {  0.5f,  0.0f, -0.5f },
                {  0.5f,  0.0f,  0.5f },
                { -0.5f,  0.0f, -0.5f },
                { -0.5f,  0.0f,  0.5f }
            };
            std::vector<uint3> mesh_indices;
            std::vector<uint32_t> mesh_sbt_indices;

            uint32_t mesh_index = 0;
            mesh_indices.emplace_back(make_uint3(mesh_index + 0, mesh_index + 1, mesh_index + 2));
            mesh_index += 3;
            mesh_indices.emplace_back(make_uint3(mesh_index + 0, mesh_index + 1, mesh_index + 2));
            mesh_index += 3;
	    */
	    
	    std::vector<float3> vertices;
	    std::vector<uint3> mesh_indices;
            std::vector<uint32_t> mesh_sbt_indices;

	    OptixAabb aabb;
	    aabb = read_obj_mesh(obj_file, vertices, mesh_indices);
	    
	    
            //set sbt index (only one material available)
            const uint32_t sbt_index = 0;
            for (size_t i = 0; i < mesh_indices.size(); i++) {
                mesh_sbt_indices.push_back(sbt_index);
            }

            //build mesh GAS
            GeometryAccelData mesh_gas;
            void* d_mesh_data;
            d_mesh_data = buildTriangleGAS(context, mesh_gas, vertices, mesh_indices, mesh_sbt_indices);

            //HitGroupDataを追加
            hitgroup_datas.emplace_back(ShapeType::Mesh, HitGroupData{ d_mesh_data });
            
            //Custom build input for sphere: simple list of sphere data
            std::vector<SphereData> spheres;
            for (int i = 0; i < num_sphere; i++) {

	      //define position of recieve sphere 
	      //const float3 center{ 0.0f, 2.0f, i };
	      //const float3 center{ 0.0f, -1.0f,  0.0f };        //for planes0, planes1, sphere
	      //const float3 center{ 0.0f, 0.0f, -0.1f };         //for cow
	      //const float3 center{ 0.0f, 0.0f, -5.0f };         //for SphereAndPlane
	      const float3 center{ -17998.0f, -93650.0f, 48.0f }; //for meidaiITC
	      

	      //define radious of recieve sphere
	      spheres.emplace_back(SphereData{ center, 0.04f });  //for meidaiITC, SphereAndPlene, planes0, planes1, sphere
	      //spheres.emplace_back(SphereData{ center, 0.01f }) //for cow
	      //spheres.emplace_back(SphereData{ center, 0.5f });
            }

            std::vector<uint32_t> sphere_sbt_indices;
            uint32_t sphere_sbt_index = 0;
            for (int i = 0; i < num_sphere; i++) {
                sphere_sbt_indices.push_back(sphere_sbt_index);
            }

            GeometryAccelData sphere_gas;
            void* d_sphere_data;
            d_sphere_data = buildSphereGAS(context, sphere_gas, spheres, sphere_sbt_indices);

            //HitGroupDataを追加
            hitgroup_datas.emplace_back(ShapeType::Sphere, HitGroupData{ d_sphere_data });

            // IAS用のInstanceを球体用・メッシュ用それぞれ作成
            std::vector<OptixInstance> instances;
            uint32_t flags = OPTIX_INSTANCE_FLAG_NONE;

            uint32_t sbt_offset = 0;
            uint32_t instance_id = 0;
            instances.emplace_back(OptixInstance{
                {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, instance_id, sbt_offset, 255,
                flags, mesh_gas.handle, {0, 0}
                });

            sbt_offset += sphere_gas.num_sbt_records;
            instance_id++;
            instances.push_back(OptixInstance{
                {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, instance_id, sbt_offset, 255,
                flags, sphere_gas.handle, {0, 0}
                });

            // IASの作成
            buildIAS(context, ias, instances);

        }

        //
        // Create module
        //
        OptixModule module = nullptr;
        OptixModule sphere_module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

            pipeline_compile_options.usesMotionBlur        = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
            pipeline_compile_options.numPayloadValues      = 2;
            pipeline_compile_options.numAttributeValues    = 5;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
#endif
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

            size_t      inputSize  = 0;
            std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "meidaiITC.cu");
            size_t sizeof_log = sizeof( log );

            OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                        context,
                        &module_compile_options,
                        &pipeline_compile_options,
                        ptx.c_str(),
                        ptx.size(),
                        log,
                        &sizeof_log,
                        &module
                        ) );
        }


        //
        // Create program groups
        //
        OptixProgramGroup raygen_prog_group   = nullptr;
        OptixProgramGroup miss_prog_group     = nullptr;
        OptixProgramGroup hitgroup_prog_group_triangle = nullptr;
        OptixProgramGroup hitgroup_prog_group_sphere = nullptr;
        {
            OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            size_t sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &raygen_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &raygen_prog_group
                        ) );

            OptixProgramGroupDesc miss_prog_group_desc  = {};
            miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module            = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &miss_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &miss_prog_group
                        ) );

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__triangle";
            sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &hitgroup_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &hitgroup_prog_group_triangle
                        ) );

            memset(&hitgroup_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
            hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleIS = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
            hitgroup_prog_group_desc.hitgroup.moduleCH = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &hitgroup_prog_group_sphere
            ));
        }

        //
        // Link pipeline
        //
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t    max_trace_depth  = 31;
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group_triangle, hitgroup_prog_group_sphere };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth          = max_trace_depth;
            pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            size_t sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixPipelineCreate(
                        context,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups,
                        sizeof( program_groups ) / sizeof( program_groups[0] ),
                        log,
                        &sizeof_log,
                        &pipeline
                        ) );

            OptixStackSizes stack_sizes = {};
            for( auto& prog_group : program_groups )
            {
                OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                     0,  // maxCCDepth
                                                     0,  // maxDCDEpth
                                                     &direct_callable_stack_size_from_traversal,
                                                     &direct_callable_stack_size_from_state, &continuation_stack_size ) );
            OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                                    direct_callable_stack_size_from_state, continuation_stack_size,
                                                    2  // maxTraversableDepth
                                                    ) );
        }

        //
        // Set up shader binding table
        //
        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof( RayGenSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
            RayGenSbtRecord rg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( raygen_record ),
                        &rg_sbt,
                        raygen_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof( MissSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
            MissSbtRecord ms_sbt;
            ms_sbt.data = { 0.3f, 0.1f, 0.2f };
            OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( miss_record ),
                        &ms_sbt,
                        miss_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            // HitGroup
            CUdeviceptr hitgroup_record;
            size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord ) * hitgroup_datas.size(); //triangle and sphere
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
            int hit_idx = 0;
            HitGroupSbtRecord* hg_sbt = new HitGroupSbtRecord[hitgroup_datas.size()];
            HitGroupData data = hitgroup_datas[hit_idx].second;
            OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group_triangle, &hg_sbt[ hit_idx ] ) );
            hg_sbt[hit_idx].data = data;
            hit_idx++;
            data = hitgroup_datas[hit_idx].second;
            OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group_sphere, &hg_sbt[ hit_idx ] ) );
            hg_sbt[hit_idx].data = data;
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void**>( hitgroup_record ),
                        hg_sbt,
                        hitgroup_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            sbt.raygenRecord                = raygen_record;
            sbt.missRecordBase              = miss_record;
            sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = hitgroup_record;
            sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitGroupSbtRecord));
            sbt.hitgroupRecordCount         = static_cast<uint32_t>(hitgroup_datas.size());;
        }

        sutil::CUDAOutputBuffer<uchar4> output_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height );
	//sutil::CUDAOutputBuffer<Result> result( sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height );

        //
        // launch
        //
        {
            CUstream stream;
            CUDA_CHECK( cudaStreamCreate( &stream ) );
            output_buffer.setStream(stream);
	    //result.buffer.setStream(stream);

	    // for Opal calculaton
	    /*
	    float freq = 5.9e9f;
            float wavelength = 299792458.0f / freq; ;
            float3 polarization = make_float3(0.0f, 1.0f, 0.0f); //Perpendicular to the floor. Assuming as in Unity that forward is z-axis and up is y-axis
            float k = 2 * 3.14159265358979323846f / wavelength;
            float4 polarizition_k = make_float4(polarization, k);
	    float2 dielectricConstant = make_float2(3.75f, -60.0f * wavelength * 0.038f);
	    */

	    
            sutil::Camera cam;
            configureCamera( cam, width, height );

	    /*
	    float* result;
	    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &result ), num_sphere * sizeof(float)));
	    */
	    
            Params params;	    
            //params.image        = output_buffer.map();
	    //params.result       = result.map();
            params.image_width  = width;
            params.image_height = height;
            params.handle       = ias.handle;
            params.cam_eye      = cam.eye();
            cam.UVWFrame( params.cam_u, params.cam_v, params.cam_w );

            CUdeviceptr d_param;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_param ),
                        &params, sizeof( Params ),
                        cudaMemcpyHostToDevice
                        ) );
	    Timer timer3;
	    
            OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, /*depth=*/1 ) );
            CUDA_SYNC_CHECK();
	    
	    timer3.stop();
	    
	    std::cout << std::setw(32) << std::left << "Launch took: " << std::setw(12)
	    << std::right << timer3.get_elapsed_s() << "s\n";
            //output_buffer.unmap();
	    //result.unmap();
        }

        //
        // Display results
        //
        /*
        {
            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = width;
            buffer.height       = height;
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            if( outfile.empty() )
                sutil::displayBufferWindow( argv[0], buffer );
            else
                sutil::saveImage( outfile.c_str(), buffer, false );
        }
        */
        //
        // Cleanup
        //
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer    ) ) );

            OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
            OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group_triangle ));
            OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group_sphere ) );
            OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
            OPTIX_CHECK( optixModuleDestroy( module ) );

            OPTIX_CHECK( optixDeviceContextDestroy( context ) );
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
