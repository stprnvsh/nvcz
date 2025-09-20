#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <cuda_runtime.h>

namespace nvcz {

/**
 * @brief Compression algorithms supported by nvcz
 */
enum class Algorithm {
    LZ4 = 0,
    GDEFLATE = 1,
    SNAPPY = 2,
    ZSTD = 3
};

/**
 * @brief Compression statistics for performance monitoring
 */
struct CompressionStats {
    size_t input_bytes = 0;
    size_t compressed_bytes = 0;
    double compression_ratio = 0.0;
    double throughput_mbps = 0.0;
    int gpu_count = 0;
    std::string algorithm;
};

/**
 * @brief Configuration for nvcz compression/decompression
 */
struct Config {
    Algorithm algorithm = Algorithm::LZ4;
    uint32_t chunk_mb = 32;           // Chunk size in MiB
    size_t nvcomp_chunk_kb = 64;      // nvCOMP internal chunk size in KiB
    bool enable_autotune = true;      // Enable automatic tuning
    int streams = 0;                  // Number of CUDA streams (0 = autotune)
    bool multi_gpu = false;           // Enable multi-GPU mode
    std::vector<int> gpu_ids;         // GPU IDs to use (empty = all available)
    int streams_per_gpu = 2;          // Streams per GPU in multi-GPU mode
    bool size_aware = false;          // Adjust streams based on input size
};

/**
 * @brief Result of compression/decompression operations
 */
struct Result {
    bool success = false;
    std::string error_message;
    CompressionStats stats;

    explicit operator bool() const { return success; }
};

/**
 * @brief Callback function type for streaming data processing
 */
using DataCallback = std::function<void(const uint8_t* data, size_t size)>;

/**
 * @brief Callback function type for progress reporting
 */
using ProgressCallback = std::function<void(size_t processed_bytes, size_t total_bytes)>;

/**
 * @brief Main nvcz class for GPU-accelerated compression/decompression
 */
class Compressor {
public:
    /**
     * @brief Create a new compressor instance
     * @param config Configuration for compression settings
     */
    explicit Compressor(const Config& config = Config());

    /**
     * @brief Destroy the compressor instance
     */
    ~Compressor();

    /**
     * @brief Compress data from input buffer to output buffer
     * @param input Input data buffer
     * @param input_size Size of input data
     * @param output Output buffer (must be large enough)
     * @param output_size Size of output buffer (updated with actual compressed size)
     * @return Result indicating success/failure and statistics
     */
    Result compress(const uint8_t* input, size_t input_size,
                   uint8_t* output, size_t* output_size);

    /**
     * @brief Decompress data from input buffer to output buffer
     * @param input Compressed input data buffer
     * @param input_size Size of compressed input data
     * @param output Output buffer (must be large enough)
     * @param output_size Size of output buffer (updated with actual decompressed size)
     * @return Result indicating success/failure and statistics
     */
    Result decompress(const uint8_t* input, size_t input_size,
                     uint8_t* output, size_t* output_size);

    /**
     * @brief Get maximum compressed size for given input size
     * @param input_size Size of input data
     * @return Maximum possible compressed size
     */
    size_t get_max_compressed_size(size_t input_size) const;

    /**
     * @brief Get current configuration
     * @return Copy of current configuration
     */
    Config get_config() const;

    /**
     * @brief Update configuration (requires reinitialization)
     * @param config New configuration
     * @return Result indicating success/failure
     */
    Result update_config(const Config& config);

    /**
     * @brief Get performance statistics from last operation
     * @return Statistics from the last compression/decompression
     */
    const CompressionStats& get_last_stats() const;

    // Disable copying
    Compressor(const Compressor&) = delete;
    Compressor& operator=(const Compressor&) = delete;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Convenience function to create a compressor with default settings
 * @return New compressor instance
 */
std::unique_ptr<Compressor> create_compressor();

/**
 * @brief Convenience function to create a compressor with custom config
 * @param config Configuration for compression settings
 * @return New compressor instance
 */
std::unique_ptr<Compressor> create_compressor(const Config& config);

/**
 * @brief Stream compression from input callback to output callback
 * @param config Compression configuration
 * @param input_callback Function called to get input data
 * @param output_callback Function called with compressed output data
 * @param progress_callback Optional progress reporting function
 * @return Result indicating success/failure and statistics
 */
Result compress_stream(const Config& config,
                      DataCallback input_callback,
                      DataCallback output_callback,
                      ProgressCallback progress_callback = nullptr);

/**
 * @brief Stream decompression from input callback to output callback
 * @param input_callback Function called to get compressed input data
 * @param output_callback Function called with decompressed output data
 * @param progress_callback Optional progress reporting function
 * @return Result indicating success/failure and statistics
 */
Result decompress_stream(DataCallback input_callback,
                        DataCallback output_callback,
                        ProgressCallback progress_callback = nullptr);

/**
 * @brief Get version information
 * @return String containing version information
 */
std::string get_version();

/**
 * @brief Check if CUDA and nvCOMP are available
 * @return True if all required components are available
 */
bool is_available();

/**
 * @brief Get information about available GPUs
 * @return Vector of GPU IDs that are available for compression
 */
std::vector<int> get_available_gpus();

} // namespace nvcz
