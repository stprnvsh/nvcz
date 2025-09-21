#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
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
 * @brief Callback function type for data provision in streaming
 */
using DataProvider = std::function<std::pair<const uint8_t*, size_t>()>;

/**
 * @brief Ring buffer configuration for streaming
 */
struct RingBufferConfig {
    size_t buffer_size_mb = 32;        // Size of each ring buffer in MiB
    size_t ring_slots = 3;             // Number of buffers in the ring
    bool enable_overlapped_io = true;   // Enable overlapped I/O and computation
};

/**
 * @brief Streaming statistics for performance monitoring
 */
struct StreamingStats {
    size_t total_input_bytes = 0;
    size_t total_output_bytes = 0;
    double compression_ratio = 0.0;
    double throughput_mbps = 0.0;
    size_t chunks_processed = 0;
    double io_overlap_efficiency = 0.0; // Percentage of time I/O overlapped with computation
    std::chrono::microseconds total_time;
    std::chrono::microseconds io_time;
    std::chrono::microseconds compute_time;
};

/**
 * @brief Advanced streaming interface for large file processing
 */
class StreamProcessor {
public:
    virtual ~StreamProcessor() = default;

    /**
     * @brief Initialize the stream processor
     * @param config Configuration for streaming
     * @return true on success
     */
    virtual bool initialize(const Config& config) = 0;

    /**
     * @brief Get next chunk of input data
     * @param buffer Buffer to fill with input data
     * @param buffer_size Size of buffer
     * @param bytes_read Output parameter for bytes actually read
     * @return true if more data is available, false if EOF
     */
    virtual bool get_next_input_chunk(uint8_t* buffer, size_t buffer_size, size_t* bytes_read) = 0;

    /**
     * @brief Process compressed output chunk
     * @param data Compressed data
     * @param size Size of compressed data
     * @return true on success
     */
    virtual bool process_output_chunk(const uint8_t* data, size_t size) = 0;

    /**
     * @brief Finalize the stream processor
     * @return true on success
     */
    virtual bool finalize() = 0;

    /**
     * @brief Get total bytes processed
     * @return Total input bytes processed
     */
    virtual size_t get_total_bytes_processed() const = 0;
};

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

    /**
     * @brief Compress a file with streaming for large files
     * @param input_file Path to input file
     * @param output_file Path to output file
     * @param chunk_size_mb Size of processing chunks in MiB (affects memory usage)
     * @param progress_callback Optional progress reporting function
     * @return Result indicating success/failure and streaming statistics
     */
    Result compress_file(const std::string& input_file,
                        const std::string& output_file,
                        size_t chunk_size_mb = 32,
                        ProgressCallback progress_callback = nullptr);

    /**
     * @brief Decompress a file with streaming for large files
     * @param input_file Path to compressed input file
     * @param output_file Path to decompressed output file
     * @param progress_callback Optional progress reporting function
     * @return Result indicating success/failure and streaming statistics
     */
    Result decompress_file(const std::string& input_file,
                          const std::string& output_file,
                          ProgressCallback progress_callback = nullptr);

    /**
     * @brief Advanced streaming compression with custom processor
     * @param processor Stream processor implementation
     * @param ring_config Ring buffer configuration for overlapped I/O
     * @param progress_callback Optional progress reporting function
     * @return Streaming statistics and success/failure information
     */
    StreamingStats compress_with_streaming(StreamProcessor* processor,
                                          const RingBufferConfig& ring_config = RingBufferConfig(),
                                          ProgressCallback progress_callback = nullptr);

    /**
     * @brief Advanced streaming decompression with custom processor
     * @param processor Stream processor implementation
     * @param ring_config Ring buffer configuration for overlapped I/O
     * @param progress_callback Optional progress reporting function
     * @return Streaming statistics and success/failure information
     */
    StreamingStats decompress_with_streaming(StreamProcessor* processor,
                                            const RingBufferConfig& ring_config = RingBufferConfig(),
                                            ProgressCallback progress_callback = nullptr);

    /**
     * @brief Enable multi-GPU support for subsequent operations
     * @param gpu_ids Vector of GPU IDs to use (empty = all available)
     * @param streams_per_gpu Number of CUDA streams per GPU
     * @return true on success
     */
    bool enable_multi_gpu(const std::vector<int>& gpu_ids = {},
                         int streams_per_gpu = 2);

    /**
     * @brief Get current GPU configuration
     * @return Vector of GPU IDs currently in use
     */
    std::vector<int> get_active_gpus() const;

    /**
     * @brief Get streaming statistics for monitoring performance
     * @return Detailed statistics about streaming efficiency
     */
    StreamingStats get_streaming_stats() const;

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

/**
 * @brief Convenience function for file compression with automatic streaming
 * @param input_file Path to input file
 * @param output_file Path to output file
 * @param config Compression configuration
 * @param progress_callback Optional progress reporting function
 * @return Result indicating success/failure and statistics
 */
Result compress_file(const std::string& input_file,
                    const std::string& output_file,
                    const Config& config = Config(),
                    ProgressCallback progress_callback = nullptr);

/**
 * @brief Convenience function for file decompression with automatic streaming
 * @param input_file Path to compressed input file
 * @param output_file Path to decompressed output file
 * @param config Decompression configuration
 * @param progress_callback Optional progress reporting function
 * @return Result indicating success/failure and statistics
 */
Result decompress_file(const std::string& input_file,
                      const std::string& output_file,
                      const Config& config = Config(),
                      ProgressCallback progress_callback = nullptr);

/**
 * @brief File-based stream processor implementation
 */
class FileStreamProcessor : public StreamProcessor {
public:
    /**
     * @brief Create a file stream processor
     * @param input_file Path to input file
     * @param output_file Path to output file
     */
    explicit FileStreamProcessor(const std::string& input_file,
                                const std::string& output_file);

    /**
     * @brief Create a file stream processor with output callback
     * @param input_file Path to input file
     * @param output_callback Callback for compressed data output
     */
    explicit FileStreamProcessor(const std::string& input_file,
                                DataCallback output_callback);

    ~FileStreamProcessor() override;

    bool initialize(const Config& config) override;
    bool get_next_input_chunk(uint8_t* buffer, size_t buffer_size, size_t* bytes_read) override;
    bool process_output_chunk(const uint8_t* data, size_t size) override;
    bool finalize() override;
    size_t get_total_bytes_processed() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Memory buffer stream processor for in-memory processing
 */
class MemoryStreamProcessor : public StreamProcessor {
public:
    /**
     * @brief Create a memory stream processor
     * @param input_data Input data buffer
     * @param input_size Size of input data
     * @param output_callback Callback for compressed output data
     */
    explicit MemoryStreamProcessor(const uint8_t* input_data, size_t input_size,
                                  DataCallback output_callback);

    ~MemoryStreamProcessor() override;

    bool initialize(const Config& config) override;
    bool get_next_input_chunk(uint8_t* buffer, size_t buffer_size, size_t* bytes_read) override;
    bool process_output_chunk(const uint8_t* data, size_t size) override;
    bool finalize() override;
    size_t get_total_bytes_processed() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Ring buffer manager for overlapped I/O operations
 */
class RingBufferManager {
public:
    /**
     * @brief Create a ring buffer manager
     * @param config Ring buffer configuration
     */
    explicit RingBufferManager(const RingBufferConfig& config = RingBufferConfig());

    ~RingBufferManager();

    /**
     * @brief Initialize ring buffers
     * @param gpu_count Number of GPUs to create buffers for
     * @return true on success
     */
    bool initialize(size_t gpu_count);

    /**
     * @brief Get next available input buffer
     * @return Pointer to input buffer and its size
     */
    std::pair<uint8_t*, size_t> get_input_buffer();

    /**
     * @brief Get next available output buffer
     * @return Pointer to output buffer and its size
     */
    std::pair<uint8_t*, size_t> get_output_buffer();

    /**
     * @brief Mark input buffer as filled
     * @param buffer Pointer to filled buffer
     * @param size Number of bytes filled
     */
    void mark_input_buffer_filled(uint8_t* buffer, size_t size);

    /**
     * @brief Mark output buffer as ready for consumption
     * @param buffer Pointer to output buffer
     * @param size Number of bytes in output buffer
     */
    void mark_output_buffer_ready(uint8_t* buffer, size_t size);

    /**
     * @brief Get efficiency statistics
     * @return Ring buffer efficiency metrics
     */
    double get_efficiency() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nvcz
