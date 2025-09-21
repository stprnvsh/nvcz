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
#include <fstream>
#include <functional>
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

    // File-based operations
    Result compress_file_internal(const std::string& input_file,
                                 const std::string& output_file,
                                 size_t chunk_size_mb,
                                 ProgressCallback progress_callback);

    Result decompress_file_internal(const std::string& input_file,
                                   const std::string& output_file,
                                   ProgressCallback progress_callback);

    // Streaming operations
    StreamingStats compress_with_streaming_internal(StreamProcessor* processor,
                                                  const RingBufferConfig& ring_config,
                                                  ProgressCallback progress_callback);

    StreamingStats decompress_with_streaming_internal(StreamProcessor* processor,
                                                     const RingBufferConfig& ring_config,
                                                     ProgressCallback progress_callback);

    // Multi-GPU operations
    bool enable_multi_gpu_internal(const std::vector<int>& gpu_ids, int streams_per_gpu);
    std::vector<int> get_active_gpus_internal() const;
    StreamingStats get_streaming_stats_internal() const;

private:
    Config config_;
    CompressionStats stats_;
    StreamingStats streaming_stats_;

    // Multi-GPU state
    bool multi_gpu_enabled_ = false;
    std::vector<int> active_gpu_ids_;
    int streams_per_gpu_ = 2;

    // Ring buffer state for streaming
    std::unique_ptr<RingBufferManager> ring_buffer_manager_;

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

    Result compress_file_internal(const std::string& input_file,
                                 const std::string& output_file,
                                 size_t chunk_size_mb,
                                 ProgressCallback progress_callback) {
        Result result;

        try {
            // Open input file
            std::ifstream input(input_file, std::ios::binary);
            if (!input) {
                result.success = false;
                result.error_message = "Failed to open input file: " + input_file;
                return result;
            }

            // Open output file
            std::ofstream output(output_file, std::ios::binary);
            if (!output) {
                result.success = false;
                result.error_message = "Failed to open output file: " + output_file;
                return result;
            }

            // Get file size for progress reporting
            input.seekg(0, std::ios::end);
            size_t total_size = input.tellg();
            input.seekg(0, std::ios::beg);

            // For large files, use streaming approach
            if (total_size > chunk_size_mb * 1024 * 1024) {
                // Use chunked approach for large files
                const size_t chunk_size = chunk_size_mb * 1024 * 1024;
                std::vector<uint8_t> input_buffer(chunk_size);
                std::vector<uint8_t> output_buffer;

                size_t processed = 0;
                while (processed < total_size) {
                    // Read chunk
                    size_t to_read = std::min(chunk_size, total_size - processed);
                    input.read(reinterpret_cast<char*>(input_buffer.data()), to_read);
                    size_t actually_read = input.gcount();

                    if (actually_read == 0) break;

                    // Compress chunk
                    output_buffer.resize(get_max_compressed_size(actually_read));
                    size_t compressed_size = output_buffer.size();

                    Result chunk_result = compress_single_gpu(
                        input_buffer.data(), actually_read,
                        output_buffer.data(), &compressed_size
                    );

                    if (!chunk_result) {
                        result.success = false;
                        result.error_message = "Compression failed: " + chunk_result.error_message;
                        return result;
                    }

                    // Write compressed data
                    output.write(reinterpret_cast<const char*>(output_buffer.data()), compressed_size);

                    processed += actually_read;

                    // Progress callback
                    if (progress_callback) {
                        progress_callback(processed, total_size);
                    }
                }
            } else {
                // For smaller files, read all at once for simplicity
                std::vector<uint8_t> input_buffer(total_size);
                input.read(reinterpret_cast<char*>(input_buffer.data()), total_size);
                size_t actually_read = input.gcount();

                std::vector<uint8_t> output_buffer(get_max_compressed_size(actually_read));
                size_t compressed_size = output_buffer.size();

                Result chunk_result = compress_single_gpu(
                    input_buffer.data(), actually_read,
                    output_buffer.data(), &compressed_size
                );

                if (!chunk_result) {
                    result.success = false;
                    result.error_message = "Compression failed: " + chunk_result.error_message;
                    return result;
                }

                output.write(reinterpret_cast<const char*>(output_buffer.data()), compressed_size);

                if (progress_callback) {
                    progress_callback(actually_read, actually_read);
                }
            }

            result.success = true;
            result.stats = stats_;

        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = std::string("File compression failed: ") + e.what();
        }

        return result;
    }

    Result decompress_file_internal(const std::string& input_file,
                                   const std::string& output_file,
                                   ProgressCallback progress_callback) {
        Result result;

        try {
            // Open input file
            std::ifstream input(input_file, std::ios::binary);
            if (!input) {
                result.success = false;
                result.error_message = "Failed to open input file: " + input_file;
                return result;
            }

            // Open output file
            std::ofstream output(output_file, std::ios::binary);
            if (!output) {
                result.success = false;
                result.error_message = "Failed to open output file: " + output_file;
                return result;
            }

            // Get input file size
            input.seekg(0, std::ios::end);
            size_t input_size = input.tellg();
            input.seekg(0, std::ios::beg);

            // For large files, use chunked approach
            const size_t buffer_size = 64 * 1024 * 1024; // 64MB buffer
            std::vector<uint8_t> input_buffer(buffer_size);
            std::vector<uint8_t> output_buffer(buffer_size * 2); // Conservative estimate

            size_t processed = 0;
            while (processed < input_size) {
                // Read chunk
                size_t to_read = std::min(buffer_size, input_size - processed);
                input.read(reinterpret_cast<char*>(input_buffer.data()), to_read);
                size_t actually_read = input.gcount();

                if (actually_read == 0) break;

                // Decompress chunk
                size_t decompressed_size = output_buffer.size();
                Result chunk_result = decompress_single_gpu(
                    input_buffer.data(), actually_read,
                    output_buffer.data(), &decompressed_size
                );

                if (!chunk_result) {
                    result.success = false;
                    result.error_message = "Decompression failed: " + chunk_result.error_message;
                    return result;
                }

                // Write decompressed data
                output.write(reinterpret_cast<const char*>(output_buffer.data()), decompressed_size);

                processed += actually_read;

                // Progress callback
                if (progress_callback) {
                    progress_callback(processed, input_size);
                }
            }

            result.success = true;
            result.stats = stats_;

        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = std::string("File decompression failed: ") + e.what();
        }

        return result;
    }

    StreamingStats compress_with_streaming_internal(StreamProcessor* processor,
                                                  const RingBufferConfig& ring_config,
                                                  ProgressCallback progress_callback) {
        StreamingStats stats;
        auto start_time = std::chrono::high_resolution_clock::now();

        try {
            if (!processor) {
                stats.total_time = std::chrono::microseconds(0);
                return stats;
            }

            // Initialize the processor
            if (!processor->initialize(config_)) {
                stats.total_time = std::chrono::microseconds(0);
                return stats;
            }

            // Create ring buffer for streaming
            RingBufferManager ring_buffer(ring_config);
            if (!ring_buffer.initialize(1)) { // Single GPU for now
                stats.total_time = std::chrono::microseconds(0);
                return stats;
            }

            size_t total_processed = 0;
            size_t chunks_processed = 0;
            bool has_more_data = true;

            while (has_more_data) {
                // Get input chunk from processor
                auto [buffer, buffer_size] = ring_buffer.get_input_buffer();
                size_t bytes_read = 0;

                has_more_data = processor->get_next_input_chunk(
                    reinterpret_cast<uint8_t*>(buffer),
                    buffer_size,
                    &bytes_read
                );

                if (bytes_read == 0 && !has_more_data) break;

                // Mark buffer as filled
                ring_buffer.mark_input_buffer_filled(
                    reinterpret_cast<uint8_t*>(buffer),
                    bytes_read
                );

                // Process the chunk
                size_t compressed_size = 0;
                Result result = compress_single_gpu(
                    reinterpret_cast<uint8_t*>(buffer),
                    bytes_read,
                    reinterpret_cast<uint8_t*>(buffer), // Use same buffer for simplicity
                    &compressed_size
                );

                if (!result) {
                    stats.total_time = std::chrono::microseconds(0);
                    return stats;
                }

                // Get output buffer and process it
                auto [output_buffer, output_buffer_size] = ring_buffer.get_output_buffer();
                if (compressed_size <= output_buffer_size) {
                    std::memcpy(output_buffer, buffer, compressed_size);
                    ring_buffer.mark_output_buffer_ready(
                        reinterpret_cast<uint8_t*>(output_buffer),
                        compressed_size
                    );

                    // Send to processor
                    if (!processor->process_output_chunk(
                        reinterpret_cast<uint8_t*>(output_buffer),
                        compressed_size
                    )) {
                        stats.total_time = std::chrono::microseconds(0);
                        return stats;
                    }
                }

                total_processed += bytes_read;
                chunks_processed++;

                // Progress callback
                if (progress_callback) {
                    progress_callback(total_processed, total_processed + 1000); // Simplified progress
                }
            }

            // Finalize processor
            processor->finalize();

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

            stats.total_input_bytes = total_processed;
            stats.chunks_processed = chunks_processed;
            stats.total_time = duration;

        } catch (const std::exception& e) {
            stats.total_time = std::chrono::microseconds(0);
        }

        return stats;
    }

    StreamingStats decompress_with_streaming_internal(StreamProcessor* processor,
                                                     const RingBufferConfig& ring_config,
                                                     ProgressCallback progress_callback) {
        StreamingStats stats;
        auto start_time = std::chrono::high_resolution_clock::now();

        try {
            if (!processor) {
                stats.total_time = std::chrono::microseconds(0);
                return stats;
            }

            // Initialize the processor
            if (!processor->initialize(config_)) {
                stats.total_time = std::chrono::microseconds(0);
                return stats;
            }

            // Create ring buffer for streaming
            RingBufferManager ring_buffer(ring_config);
            if (!ring_buffer.initialize(1)) { // Single GPU for now
                stats.total_time = std::chrono::microseconds(0);
                return stats;
            }

            size_t total_processed = 0;
            size_t chunks_processed = 0;
            bool has_more_data = true;

            while (has_more_data) {
                // Get input chunk from processor
                auto [buffer, buffer_size] = ring_buffer.get_input_buffer();
                size_t bytes_read = 0;

                has_more_data = processor->get_next_input_chunk(
                    reinterpret_cast<uint8_t*>(buffer),
                    buffer_size,
                    &bytes_read
                );

                if (bytes_read == 0 && !has_more_data) break;

                // Mark buffer as filled
                ring_buffer.mark_input_buffer_filled(
                    reinterpret_cast<uint8_t*>(buffer),
                    bytes_read
                );

                // Process the chunk (decompress)
                size_t decompressed_size = buffer_size * 2; // Conservative estimate
                Result result = decompress_single_gpu(
                    reinterpret_cast<uint8_t*>(buffer),
                    bytes_read,
                    reinterpret_cast<uint8_t*>(buffer),
                    &decompressed_size
                );

                if (!result) {
                    stats.total_time = std::chrono::microseconds(0);
                    return stats;
                }

                // Get output buffer and process it
                auto [output_buffer, output_buffer_size] = ring_buffer.get_output_buffer();
                if (decompressed_size <= output_buffer_size) {
                    std::memcpy(output_buffer, buffer, decompressed_size);
                    ring_buffer.mark_output_buffer_ready(
                        reinterpret_cast<uint8_t*>(output_buffer),
                        decompressed_size
                    );

                    // Send to processor
                    if (!processor->process_output_chunk(
                        reinterpret_cast<uint8_t*>(output_buffer),
                        decompressed_size
                    )) {
                        stats.total_time = std::chrono::microseconds(0);
                        return stats;
                    }
                }

                total_processed += bytes_read;
                chunks_processed++;

                // Progress callback
                if (progress_callback) {
                    progress_callback(total_processed, total_processed + 1000); // Simplified progress
                }
            }

            // Finalize processor
            processor->finalize();

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

            stats.total_input_bytes = total_processed;
            stats.chunks_processed = chunks_processed;
            stats.total_time = duration;

        } catch (const std::exception& e) {
            stats.total_time = std::chrono::microseconds(0);
        }

        return stats;
    }

    bool enable_multi_gpu_internal(const std::vector<int>& gpu_ids, int streams_per_gpu) {
        try {
            // Validate GPU IDs
            int device_count = 0;
            cudaGetDeviceCount(&device_count);

            std::vector<int> ids_to_use = gpu_ids;
            if (ids_to_use.empty()) {
                for (int i = 0; i < device_count; ++i) {
                    ids_to_use.push_back(i);
                }
            }

            // Validate all GPU IDs
            for (int gpu_id : ids_to_use) {
                if (gpu_id < 0 || gpu_id >= device_count) {
                    return false;
                }
            }

            active_gpu_ids_ = ids_to_use;
            streams_per_gpu_ = streams_per_gpu;
            multi_gpu_enabled_ = true;

            // Initialize ring buffer manager for multi-GPU streaming
            ring_buffer_manager_ = std::make_unique<RingBufferManager>(RingBufferConfig{
                .buffer_size_mb = config_.chunk_mb,
                .ring_slots = static_cast<size_t>(streams_per_gpu * 2),
                .enable_overlapped_io = true
            });

            ring_buffer_manager_->initialize(ids_to_use.size());

            return true;

        } catch (const std::exception& e) {
            return false;
        }
    }

    std::vector<int> get_active_gpus_internal() const {
        return active_gpu_ids_;
    }

    StreamingStats get_streaming_stats_internal() const {
        return streaming_stats_;
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

// Implementation of new public API methods
Result Compressor::compress_file(const std::string& input_file,
                                const std::string& output_file,
                                size_t chunk_size_mb,
                                ProgressCallback progress_callback) {
    return impl_->compress_file_internal(input_file, output_file, chunk_size_mb, progress_callback);
}

Result Compressor::decompress_file(const std::string& input_file,
                                  const std::string& output_file,
                                  ProgressCallback progress_callback) {
    return impl_->decompress_file_internal(input_file, output_file, progress_callback);
}

StreamingStats Compressor::compress_with_streaming(StreamProcessor* processor,
                                                 const RingBufferConfig& ring_config,
                                                 ProgressCallback progress_callback) {
    return impl_->compress_with_streaming_internal(processor, ring_config, progress_callback);
}

StreamingStats Compressor::decompress_with_streaming(StreamProcessor* processor,
                                                    const RingBufferConfig& ring_config,
                                                    ProgressCallback progress_callback) {
    return impl_->decompress_with_streaming_internal(processor, ring_config, progress_callback);
}

bool Compressor::enable_multi_gpu(const std::vector<int>& gpu_ids, int streams_per_gpu) {
    return impl_->enable_multi_gpu_internal(gpu_ids, streams_per_gpu);
}

std::vector<int> Compressor::get_active_gpus() const {
    return impl_->get_active_gpus_internal();
}

StreamingStats Compressor::get_streaming_stats() const {
    return impl_->get_streaming_stats_internal();
}

// Implementation of standalone functions
Result compress_file(const std::string& input_file,
                    const std::string& output_file,
                    const Config& config,
                    ProgressCallback progress_callback) {
    Compressor compressor(config);
    return compressor.compress_file(input_file, output_file, config.chunk_mb, progress_callback);
}

Result decompress_file(const std::string& input_file,
                      const std::string& output_file,
                      const Config& config,
                      ProgressCallback progress_callback) {
    Compressor compressor(config);
    return compressor.decompress_file(input_file, output_file, progress_callback);
}

// RingBufferManager implementation
class RingBufferManager::Impl {
public:
    RingBufferConfig config_;
    std::vector<uint8_t*> input_buffers_;
    std::vector<uint8_t*> output_buffers_;
    size_t current_input_index_ = 0;
    size_t current_output_index_ = 0;
    bool initialized_ = false;

    Impl(const RingBufferConfig& config) : config_(config) {}

    bool initialize(size_t gpu_count) {
        if (initialized_) return true;

        size_t buffer_size = config_.buffer_size_mb * 1024 * 1024;

        try {
            // Allocate input buffers
            input_buffers_.resize(config_.ring_slots);
            for (size_t i = 0; i < config_.ring_slots; ++i) {
                input_buffers_[i] = new uint8_t[buffer_size];
            }

            // Allocate output buffers
            output_buffers_.resize(config_.ring_slots);
            for (size_t i = 0; i < config_.ring_slots; ++i) {
                output_buffers_[i] = new uint8_t[buffer_size * 2]; // Conservative for compressed data
            }

            initialized_ = true;
            return true;

        } catch (const std::exception& e) {
            cleanup();
            return false;
        }
    }

    ~Impl() {
        cleanup();
    }

    std::pair<uint8_t*, size_t> get_input_buffer() {
        if (!initialized_) return {nullptr, 0};

        uint8_t* buffer = input_buffers_[current_input_index_];
        size_t buffer_size = config_.buffer_size_mb * 1024 * 1024;

        current_input_index_ = (current_input_index_ + 1) % config_.ring_slots;
        return {buffer, buffer_size};
    }

    std::pair<uint8_t*, size_t> get_output_buffer() {
        if (!initialized_) return {nullptr, 0};

        uint8_t* buffer = output_buffers_[current_output_index_];
        size_t buffer_size = config_.buffer_size_mb * 1024 * 1024 * 2; // Conservative

        current_output_index_ = (current_output_index_ + 1) % config_.ring_slots;
        return {buffer, buffer_size};
    }

    void mark_input_buffer_filled(uint8_t* buffer, size_t size) {
        // In a real implementation, this would coordinate with GPU operations
        // For now, just a placeholder
    }

    void mark_output_buffer_ready(uint8_t* buffer, size_t size) {
        // In a real implementation, this would coordinate with GPU operations
        // For now, just a placeholder
    }

    double get_efficiency() const {
        // Simplified efficiency metric
        return initialized_ ? 0.8 : 0.0;
    }

private:
    void cleanup() {
        for (auto* buffer : input_buffers_) {
            delete[] buffer;
        }
        input_buffers_.clear();

        for (auto* buffer : output_buffers_) {
            delete[] buffer;
        }
        output_buffers_.clear();

        initialized_ = false;
    }
};

RingBufferManager::RingBufferManager(const RingBufferConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

RingBufferManager::~RingBufferManager() = default;

bool RingBufferManager::initialize(size_t gpu_count) {
    return impl_->initialize(gpu_count);
}

std::pair<uint8_t*, size_t> RingBufferManager::get_input_buffer() {
    return impl_->get_input_buffer();
}

std::pair<uint8_t*, size_t> RingBufferManager::get_output_buffer() {
    return impl_->get_output_buffer();
}

void RingBufferManager::mark_input_buffer_filled(uint8_t* buffer, size_t size) {
    impl_->mark_input_buffer_filled(buffer, size);
}

void RingBufferManager::mark_output_buffer_ready(uint8_t* buffer, size_t size) {
    impl_->mark_output_buffer_ready(buffer, size);
}

double RingBufferManager::get_efficiency() const {
    return impl_->get_efficiency();
}

// FileStreamProcessor implementation
class FileStreamProcessor::Impl {
public:
    std::string input_file_;
    std::string output_file_;
    std::unique_ptr<std::ifstream> input_stream_;
    std::unique_ptr<std::ofstream> output_stream_;
    std::function<void(const uint8_t*, size_t)> output_callback_;
    size_t total_processed_ = 0;

    Impl(const std::string& input_file, const std::string& output_file)
        : input_file_(input_file), output_file_(output_file) {}

    Impl(const std::string& input_file, std::function<void(const uint8_t*, size_t)> output_callback)
        : input_file_(input_file), output_callback_(output_callback) {}

    bool initialize(const Config& config) {
        try {
            if (!input_file_.empty()) {
                input_stream_ = std::make_unique<std::ifstream>(input_file_, std::ios::binary);
                if (!input_stream_->is_open()) {
                    return false;
                }
            }

            if (!output_file_.empty()) {
                output_stream_ = std::make_unique<std::ofstream>(output_file_, std::ios::binary);
                if (!output_stream_->is_open()) {
                    return false;
                }
            }

            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }

    bool get_next_input_chunk(uint8_t* buffer, size_t buffer_size, size_t* bytes_read) {
        if (input_stream_ && input_stream_->is_open()) {
            input_stream_->read(reinterpret_cast<char*>(buffer), buffer_size);
            *bytes_read = input_stream_->gcount();
            return *bytes_read > 0 || input_stream_->eof();
        }
        return false;
    }

    bool process_output_chunk(const uint8_t* data, size_t size) {
        if (output_stream_ && output_stream_->is_open()) {
            output_stream_->write(reinterpret_cast<const char*>(data), size);
            return output_stream_->good();
        } else if (output_callback_) {
            output_callback_(data, size);
            return true;
        }
        return false;
    }

    bool finalize() {
        if (input_stream_) {
            input_stream_->close();
        }
        if (output_stream_) {
            output_stream_->close();
        }
        return true;
    }

    size_t get_total_bytes_processed() const {
        return total_processed_;
    }
};

FileStreamProcessor::FileStreamProcessor(const std::string& input_file, const std::string& output_file)
    : impl_(std::make_unique<Impl>(input_file, output_file)) {}

FileStreamProcessor::FileStreamProcessor(const std::string& input_file, DataCallback output_callback)
    : impl_(std::make_unique<Impl>(input_file, output_callback)) {}

FileStreamProcessor::~FileStreamProcessor() = default;

bool FileStreamProcessor::initialize(const Config& config) {
    return impl_->initialize(config);
}

bool FileStreamProcessor::get_next_input_chunk(uint8_t* buffer, size_t buffer_size, size_t* bytes_read) {
    return impl_->get_next_input_chunk(buffer, buffer_size, bytes_read);
}

bool FileStreamProcessor::process_output_chunk(const uint8_t* data, size_t size) {
    return impl_->process_output_chunk(data, size);
}

bool FileStreamProcessor::finalize() {
    return impl_->finalize();
}

size_t FileStreamProcessor::get_total_bytes_processed() const {
    return impl_->get_total_bytes_processed();
}

// MemoryStreamProcessor implementation
class MemoryStreamProcessor::Impl {
public:
    const uint8_t* input_data_;
    size_t input_size_;
    size_t current_position_;
    std::function<void(const uint8_t*, size_t)> output_callback_;
    size_t total_processed_ = 0;

    Impl(const uint8_t* input_data, size_t input_size, std::function<void(const uint8_t*, size_t)> output_callback)
        : input_data_(input_data), input_size_(input_size), current_position_(0), output_callback_(output_callback) {}

    bool initialize(const Config& config) {
        current_position_ = 0;
        return true;
    }

    bool get_next_input_chunk(uint8_t* buffer, size_t buffer_size, size_t* bytes_read) {
        if (current_position_ >= input_size_) {
            *bytes_read = 0;
            return false;
        }

        size_t to_read = std::min(buffer_size, input_size_ - current_position_);
        std::memcpy(buffer, input_data_ + current_position_, to_read);
        current_position_ += to_read;
        *bytes_read = to_read;

        total_processed_ += to_read;
        return true;
    }

    bool process_output_chunk(const uint8_t* data, size_t size) {
        if (output_callback_) {
            output_callback_(data, size);
            return true;
        }
        return false;
    }

    bool finalize() {
        return true;
    }

    size_t get_total_bytes_processed() const {
        return total_processed_;
    }
};

MemoryStreamProcessor::MemoryStreamProcessor(const uint8_t* input_data, size_t input_size, DataCallback output_callback)
    : impl_(std::make_unique<Impl>(input_data, input_size, output_callback)) {}

MemoryStreamProcessor::~MemoryStreamProcessor() = default;

bool MemoryStreamProcessor::initialize(const Config& config) {
    return impl_->initialize(config);
}

bool MemoryStreamProcessor::get_next_input_chunk(uint8_t* buffer, size_t buffer_size, size_t* bytes_read) {
    return impl_->get_next_input_chunk(buffer, buffer_size, bytes_read);
}

bool MemoryStreamProcessor::process_output_chunk(const uint8_t* data, size_t size) {
    return impl_->process_output_chunk(data, size);
}

bool MemoryStreamProcessor::finalize() {
    return impl_->finalize();
}

size_t MemoryStreamProcessor::get_total_bytes_processed() const {
    return impl_->get_total_bytes_processed();
}

} // namespace nvcz
