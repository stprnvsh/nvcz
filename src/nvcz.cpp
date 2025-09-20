#include "nvcz/nvcz.hpp"
#include "nvcz/codec.hpp"
#include "nvcz/mgpu.hpp"
#include "nvcz/autotune.hpp"
#include "nvcz/util.hpp"
#include "nvcz/framing.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <nvcomp/shared_types.h>

namespace nvcz {

namespace {

std::string get_algorithm_name(Algorithm algo) {
    switch (algo) {
        case Algorithm::LZ4: return "LZ4";
        case Algorithm::GDEFLATE: return "GDeflate";
        case Algorithm::SNAPPY: return "Snappy";
        case Algorithm::ZSTD: return "Zstd";
        default: return "Unknown";
    }
}

} // anonymous namespace

class Compressor::Impl {
public:
    explicit Impl(const Config& config) : config_(config) {
        initialize();
    }

    ~Impl() = default;

    Result compress(const uint8_t* input, size_t input_size,
                   uint8_t* output, size_t* output_size) {
        Result result;
        auto start_time = std::chrono::high_resolution_clock::now();

        try {
            if (config_.multi_gpu) {
                // Use multi-GPU path
                MgpuTune mgpu_tune = prepare_mgpu_tune();
                // Note: For library usage, we need to adapt the mgpu functions
                // This would require refactoring the mgpu code to work with callbacks
                result.success = false;
                result.error_message = "Multi-GPU streaming not yet implemented in library API";
                return result;
            } else {
                // Use single-GPU path
                result = compress_single_gpu(input, input_size, output, output_size);
            }
        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = std::string("Compression failed: ") + e.what();
            return result;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        if (result.success) {
            stats_.input_bytes = input_size;
            stats_.compressed_bytes = *output_size;
            stats_.compression_ratio = static_cast<double>(input_size) / *output_size;
            stats_.throughput_mbps = (input_size * 8.0) / (duration.count() / 1000000.0) / (1024 * 1024);
            stats_.algorithm = get_algorithm_name(config_.algorithm);
            stats_.gpu_count = config_.multi_gpu ? static_cast<int>(config_.gpu_ids.size()) : 1;
        }

        return result;
    }

    Result decompress(const uint8_t* input, size_t input_size,
                     uint8_t* output, size_t* output_size) {
        Result result;
        auto start_time = std::chrono::high_resolution_clock::now();

        try {
            if (config_.multi_gpu) {
                result.success = false;
                result.error_message = "Multi-GPU decompression not yet implemented in library API";
                return result;
            } else {
                result = decompress_single_gpu(input, input_size, output, output_size);
            }
        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = std::string("Decompression failed: ") + e.what();
            return result;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        if (result.success) {
            stats_.input_bytes = input_size;
            stats_.compressed_bytes = *output_size;
            stats_.compression_ratio = static_cast<double>(input_size) / *output_size;
            stats_.throughput_mbps = (input_size * 8.0) / (duration.count() / 1000000.0) / (1024 * 1024);
            stats_.algorithm = get_algorithm_name(config_.algorithm);
            stats_.gpu_count = config_.multi_gpu ? static_cast<int>(config_.gpu_ids.size()) : 1;
        }

        return result;
    }

    size_t get_max_compressed_size(size_t input_size) const {
        auto codec = make_codec(config_.algorithm, config_.nvcomp_chunk_kb);
        return codec ? codec->max_compressed_bound(input_size) : 0;
    }

    const CompressionStats& get_last_stats() const {
        return stats_;
    }

private:
    Config config_;
    CompressionStats stats_;

    void initialize() {
        // Validate configuration
        if (config_.algorithm < Algorithm::LZ4 || config_.algorithm > Algorithm::ZSTD) {
            throw std::invalid_argument("Invalid algorithm");
        }

        if (config_.chunk_mb == 0 || config_.chunk_mb > 1024) {
            throw std::invalid_argument("Invalid chunk size (must be 1-1024 MiB)");
        }

        if (config_.nvcomp_chunk_kb == 0 || config_.nvcomp_chunk_kb > 1024) {
            throw std::invalid_argument("Invalid nvCOMP chunk size (must be 1-1024 KiB)");
        }

        // Initialize CUDA
        int device_count = 0;
        cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("CUDA not available or no devices found");
        }

        if (device_count == 0) {
            throw std::runtime_error("No CUDA devices found");
        }

        // Validate GPU IDs if specified
        if (!config_.gpu_ids.empty()) {
            for (int gpu_id : config_.gpu_ids) {
                if (gpu_id < 0 || gpu_id >= device_count) {
                    throw std::invalid_argument("Invalid GPU ID: " + std::to_string(gpu_id));
                }
            }
        }
    }

    MgpuTune prepare_mgpu_tune() {
        MgpuTune tune;
        tune.chunk_mb = config_.chunk_mb;
        tune.streams_per_gpu = config_.streams_per_gpu;
        tune.size_aware = config_.size_aware;

        if (config_.gpu_ids.empty()) {
            tune.gpu_ids = discover_gpus_ids();
        } else {
            tune.gpu_ids = config_.gpu_ids;
        }

        return tune;
    }

    Result compress_single_gpu(const uint8_t* input, size_t input_size,
                              uint8_t* output, size_t* output_size) {
        Result result;

        // For now, use a simplified approach - in a full implementation,
        // this would use the same streaming logic as the CLI but with callbacks
        auto codec = make_codec(config_.algorithm, config_.nvcomp_chunk_kb);
        if (!codec) {
            result.success = false;
            result.error_message = "Failed to create codec";
            return result;
        }

        const size_t max_compressed = codec->max_compressed_bound(input_size);
        if (max_compressed > *output_size) {
            result.success = false;
            result.error_message = "Output buffer too small";
            return result;
        }

        // Create CUDA stream
        cudaStream_t stream;
        cudaError_t cuda_status = cudaStreamCreate(&stream);
        if (cuda_status != cudaSuccess) {
            result.success = false;
            result.error_message = "Failed to create CUDA stream";
            return result;
        }

        // Allocate device memory for size
        size_t* d_compressed_size = nullptr;
        cuda_status = cudaMallocAsync(&d_compressed_size, sizeof(size_t), stream);
        if (cuda_status != cudaSuccess) {
            cudaStreamDestroy(stream);
            result.success = false;
            result.error_message = "Failed to allocate device memory";
            return result;
        }

        // Perform compression
        codec->compress_with_stream(input, input_size, output, stream, d_compressed_size);

        // Copy size back to host
        cuda_status = cudaMemcpyAsync(&result.stats.compressed_bytes, d_compressed_size,
                                     sizeof(size_t), cudaMemcpyDeviceToHost, stream);
        if (cuda_status != cudaSuccess) {
            cudaFreeAsync(d_compressed_size, stream);
            cudaStreamDestroy(stream);
            result.success = false;
            result.error_message = "Failed to copy compressed size";
            return result;
        }

        // Synchronize and cleanup
        cuda_status = cudaStreamSynchronize(stream);
        if (cuda_status != cudaSuccess) {
            cudaFreeAsync(d_compressed_size, stream);
            cudaStreamDestroy(stream);
            result.success = false;
            result.error_message = "CUDA stream synchronization failed";
            return result;
        }

        *output_size = result.stats.compressed_bytes;
        result.success = true;

        cudaFreeAsync(d_compressed_size, stream);
        cudaStreamDestroy(stream);

        return result;
    }

    Result decompress_single_gpu(const uint8_t* input, size_t input_size,
                                uint8_t* output, size_t* output_size) {
        Result result;

        // For now, use a simplified approach - in a full implementation,
        // this would handle the framing format
        auto codec = make_codec(config_.algorithm, config_.nvcomp_chunk_kb);
        if (!codec) {
            result.success = false;
            result.error_message = "Failed to create codec";
            return result;
        }

        // Create CUDA stream
        cudaStream_t stream;
        cudaError_t cuda_status = cudaStreamCreate(&stream);
        if (cuda_status != cudaSuccess) {
            result.success = false;
            result.error_message = "Failed to create CUDA stream";
            return result;
        }

        // Perform decompression
        codec->decompress_with_stream(input, input_size, output, *output_size, stream);

        // Synchronize
        cuda_status = cudaStreamSynchronize(stream);
        if (cuda_status != cudaSuccess) {
            cudaStreamDestroy(stream);
            result.success = false;
            result.error_message = "CUDA stream synchronization failed";
            return result;
        }

        result.success = true;
        cudaStreamDestroy(stream);

        return result;
    }
};

// Implementation of public API functions

Compressor::Compressor(const Config& config) : impl_(std::make_unique<Impl>(config)) {}
Compressor::~Compressor() = default;

Result Compressor::compress(const uint8_t* input, size_t input_size,
                           uint8_t* output, size_t* output_size) {
    return impl_->compress(input, input_size, output, output_size);
}

Result Compressor::decompress(const uint8_t* input, size_t input_size,
                             uint8_t* output, size_t* output_size) {
    return impl_->decompress(input, input_size, output, output_size);
}

size_t Compressor::get_max_compressed_size(size_t input_size) const {
    return impl_->get_max_compressed_size(input_size);
}

Config Compressor::get_config() const {
    return impl_->get_config();
}

Result Compressor::update_config(const Config& config) {
    return impl_->update_config(config);
}

const CompressionStats& Compressor::get_last_stats() const {
    return impl_->get_last_stats();
}

std::unique_ptr<Compressor> create_compressor() {
    return std::make_unique<Compressor>();
}

std::unique_ptr<Compressor> create_compressor(const Config& config) {
    return std::make_unique<Compressor>(config);
}

Result compress_stream(const Config& config, DataCallback input_callback,
                      DataCallback output_callback, ProgressCallback progress_callback) {
    Result result;
    result.success = false;
    result.error_message = "Stream compression not yet implemented";
    return result;
}

Result decompress_stream(DataCallback input_callback, DataCallback output_callback,
                        ProgressCallback progress_callback) {
    Result result;
    result.success = false;
    result.error_message = "Stream decompression not yet implemented";
    return result;
}

std::string get_version() {
    return "nvcz 1.0.0 (Library API)";
}

bool is_available() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    return (status == cudaSuccess && device_count > 0);
}

std::vector<int> get_available_gpus() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        return {};
    }

    std::vector<int> gpus(device_count);
    for (int i = 0; i < device_count; ++i) {
        gpus[i] = i;
    }
    return gpus;
}

} // namespace nvcz
