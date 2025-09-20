nvcz - High-Performance GPU-Accelerated Compression
====================================================

nvcz is a command-line tool for high-performance compression and decompression using NVIDIA GPUs and the nvCOMP library. It supports multiple compression algorithms with both single-GPU and multi-GPU operation.

Features
--------

- **GPU Acceleration**: Uses NVIDIA nvCOMP library for GPU-accelerated compression
- **Multiple Algorithms**: Supports LZ4, GDeflate, Snappy, and Zstd
- **Multi-GPU Support**: Can utilize multiple GPUs for even higher throughput
- **Streaming Design**: Optimized for streaming data with overlapped I/O and computation
- **Autotuning**: Automatically optimizes chunk size and stream count based on GPU capabilities
- **Custom Format**: Simple binary format with headers for algorithm and chunk size information

Installation
------------

### Prerequisites

- CUDA Toolkit (version 11.0 or later)
- nvCOMP library (headers and libraries)
- C++17 compatible compiler (g++ recommended)
- NVIDIA GPU with sufficient memory

### Building

```bash
make
```

The build requires CUDA and nvCOMP to be installed and accessible. By default, the Makefile looks for:
- CUDA headers in `/opt/cuda/include`
- CUDA libraries in `/opt/cuda/lib64`
- nvCOMP headers in `/usr/include`
- nvCOMP libraries in `/usr/lib`

You can override these paths:
```bash
make CUDA_HOME=/path/to/cuda NVCOMP_INC=/path/to/nvcomp/include NVCOMP_LIB=/path/to/nvcomp/lib
```

### Installation

After building, install nvcz to make it available system-wide:

```bash
make install          # Install to /usr/local/bin (default)
sudo make install     # Install to /usr/local/bin with root permissions
```

Custom installation paths:
```bash
make PREFIX=/usr install      # Install to /usr/bin
make BINDIR=/opt/bin install  # Install to /opt/bin
```

Uninstall:
```bash
make uninstall        # Remove from installation directory
```

### Help

View available make targets:
```bash
make help
```

## Library Usage

nvcz can be used as a C++ library in your own applications, providing GPU-accelerated compression/decompression functionality. This allows you to integrate nvcz into custom data processing pipelines, applications, and frameworks.

### Building the Library

Build the library components:
```bash
make lib          # Build both shared and static libraries
make lib-shared   # Build shared library only
make lib-static   # Build static library only
```

### Installation

Install the library and headers system-wide:
```bash
make install-lib  # Install library and headers
sudo make install-lib  # Install with root permissions
```

Custom installation paths:
```bash
make LIBDIR=/opt/lib INCDIR=/opt/include install-lib
```

### Using as a Library

#### Basic Usage

```cpp
#include <nvcz/nvcz.hpp>
#include <iostream>
#include <vector>

int main() {
    // Check if nvcz is available
    if (!nvcz::is_available()) {
        std::cerr << "CUDA or nvCOMP not available" << std::endl;
        return 1;
    }

    // Create a compressor with default settings
    auto compressor = nvcz::create_compressor();

    // Prepare test data
    std::vector<uint8_t> input_data = {'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', '!'};
    std::vector<uint8_t> compressed_data(compressor->get_max_compressed_size(input_data.size()));

    size_t compressed_size = compressed_data.size();

    // Compress the data
    auto result = compressor->compress(
        input_data.data(),
        input_data.size(),
        compressed_data.data(),
        &compressed_size
    );

    if (result) {
        std::cout << "Compression successful!" << std::endl;
        std::cout << "Input size: " << result.stats.input_bytes << " bytes" << std::endl;
        std::cout << "Compressed size: " << result.stats.compressed_bytes << " bytes" << std::endl;
        std::cout << "Compression ratio: " << result.stats.compression_ratio << ":1" << std::endl;
        std::cout << "Throughput: " << result.stats.throughput_mbps << " MB/s" << std::endl;
    } else {
        std::cerr << "Compression failed: " << result.error_message << std::endl;
        return 1;
    }

    // Resize compressed data to actual size
    compressed_data.resize(compressed_size);

    // Decompress the data
    std::vector<uint8_t> decompressed_data(input_data.size());
    size_t decompressed_size = decompressed_data.size();

    result = compressor->decompress(
        compressed_data.data(),
        compressed_data.size(),
        decompressed_data.data(),
        &decompressed_size
    );

    if (result) {
        std::cout << "Decompression successful!" << std::endl;
        std::cout << "Decompressed data: ";
        for (auto byte : decompressed_data) {
            std::cout << static_cast<char>(byte);
        }
        std::cout << std::endl;
    }

    return 0;
}
```

#### Custom Configuration

```cpp
#include <nvcz/nvcz.hpp>

int main() {
    // Create custom configuration
    nvcz::Config config;
    config.algorithm = nvcz::Algorithm::GDEFLATE;
    config.chunk_mb = 64;           // 64 MiB chunks
    config.nvcomp_chunk_kb = 128;   // 128 KiB nvCOMP chunks
    config.multi_gpu = true;        // Enable multi-GPU
    config.gpu_ids = {0, 1};        // Use GPUs 0 and 1
    config.streams_per_gpu = 4;     // 4 streams per GPU

    // Create compressor with custom config
    auto compressor = nvcz::create_compressor(config);

    // Use compressor...
    return 0;
}
```

#### Integration with Existing Codebases

To use nvcz in your project, you'll need to:

1. **Include headers**: `#include <nvcz/nvcz.hpp>`
2. **Link the library**:
   - Shared library: `-lnvcz`
   - Static library: Path to `libnvcz.a`
3. **Set include paths**: Add nvcz include directory to compiler flags
4. **Handle CUDA dependencies**: Ensure CUDA and nvCOMP are available

**CMake Example:**
```cmake
find_package(CUDA REQUIRED)
find_library(NVCZ_LIBRARY nvcz PATHS /usr/local/lib)
find_path(NVCZ_INCLUDE_DIR nvcz/nvcz.hpp PATHS /usr/local/include)

add_executable(myapp main.cpp)
target_link_libraries(myapp ${NVCZ_LIBRARY} ${CUDA_LIBRARIES})
target_include_directories(myapp PRIVATE ${NVCZ_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
```

**Makefile Example:**
```makefile
CXX = g++
CXXFLAGS = -O2 -std=c++17 -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lnvcz -lcudart -lnvcomp -ldl

myapp: main.cpp
    $(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
```

### Library API Reference

#### Compressor Class

The main interface for compression/decompression operations:

```cpp
class Compressor {
public:
    Result compress(const uint8_t* input, size_t input_size,
                   uint8_t* output, size_t* output_size);
    Result decompress(const uint8_t* input, size_t input_size,
                     uint8_t* output, size_t* output_size);
    size_t get_max_compressed_size(size_t input_size) const;
    const CompressionStats& get_last_stats() const;
    // ... and more
};
```

#### Configuration

Control compression behavior with the `Config` struct:

```cpp
struct Config {
    Algorithm algorithm;      // LZ4, GDEFLATE, SNAPPY, or ZSTD
    uint32_t chunk_mb;        // Chunk size in MiB
    size_t nvcomp_chunk_kb;   // nvCOMP internal chunk size
    bool enable_autotune;     // Automatic performance tuning
    bool multi_gpu;           // Multi-GPU mode
    std::vector<int> gpu_ids; // GPU IDs to use
    // ... and more
};
```

#### Result and Statistics

Monitor performance and handle errors:

```cpp
struct Result {
    bool success;                    // Operation successful?
    std::string error_message;       // Error description if failed
    CompressionStats stats;          // Performance statistics
};

struct CompressionStats {
    size_t input_bytes;        // Bytes processed
    size_t compressed_bytes;   // Bytes after compression
    double compression_ratio;  // Compression ratio achieved
    double throughput_mbps;    // Processing speed in MB/s
    int gpu_count;             // Number of GPUs used
    std::string algorithm;     // Algorithm used
};
```

### Performance Considerations

- **Memory Management**: Pre-allocate output buffers using `get_max_compressed_size()`
- **Batch Processing**: For best performance, compress multiple chunks together
- **GPU Memory**: Monitor GPU memory usage, especially with large chunk sizes
- **Multi-GPU**: Use multiple GPUs for higher throughput on large datasets
- **Stream Processing**: For real-time applications, consider the streaming API (when available)

### Error Handling

The library uses RAII and provides detailed error messages:

```cpp
auto result = compressor->compress(input, input_size, output, &output_size);
if (!result) {
    std::cerr << "Error: " << result.error_message << std::endl;
    // Handle error appropriately
}
```

Usage
-----

### Compression

Basic compression with autotuning:
```bash
cat input.bin | nvcz compress --algo lz4 --auto > output.nvcz
```

Manual configuration:
```bash
cat input.bin | nvcz compress --algo gdeflate --chunk-mb 64 --streams 4 > output.nvcz
```

Multi-GPU compression:
```bash
cat input.bin | nvcz compress --algo lz4 --mgpu --gpus 0,1 > output.nvcz
```

### Decompression

Basic decompression:
```bash
nvcz decompress < input.nvcz > output.bin
```

With autotuning:
```bash
nvcz decompress --auto < input.nvcz > output.bin
```

Multi-GPU decompression:
```bash
nvcz decompress --mgpu --gpus 0,1 < input.nvcz > output.bin
```

### Command Line Options

**Compression mode:**
- `--algo {lz4|gdeflate|snappy|zstd}` - Compression algorithm
- `--chunk-mb N` - Chunk size in MiB (default: autotuned)
- `--nvcomp-chunk-kb N` - nvCOMP internal chunk size in KiB (default: 64)
- `--auto` - Enable autotuning based on GPU capabilities
- `--streams N` - Number of CUDA streams (default: autotuned)
- `--mgpu` - Enable multi-GPU mode
- `--gpus all|0,2,3` - GPU IDs to use (default: all available)
- `--streams-per-gpu N` - Streams per GPU in multi-GPU mode
- `--auto-size` - Increase stream count for large files

**Decompression mode:**
- `--auto` - Enable autotuning
- `--streams N` - Number of CUDA streams
- `--nvcomp-chunk-kb N` - nvCOMP internal chunk size in KiB (affects performance)
- `--mgpu` - Enable multi-GPU mode
- `--gpus all|0,2,3` - GPU IDs to use
- `--streams-per-gpu N` - Streams per GPU in multi-GPU mode
- `--auto-size` - Increase stream count for large files

**Note:** For decompression, `--algo` and `--chunk-mb` are ignored since the algorithm and chunk size are read from the compressed file's header.

File Format
-----------

nvcz uses a simple binary format:

```
[Header][Block1][Block2]...[BlockN][Trailer]

Header:
  - Magic: "NVCZ\\0"
  - Version: 1
  - Algorithm: 0=LZ4, 1=GDeflate, 2=Snappy, 3=Zstd
  - Chunk size in MiB

Block:
  - Raw size (8 bytes)
  - Compressed size (8 bytes)
  - Compressed data

Trailer:
  - Raw size: 0
  - Compressed size: 0
```

Architecture
------------

nvcz uses a sophisticated architecture for maximum performance:

1. **Ring Buffers**: Pre-allocated pinned host memory buffers for I/O
2. **CUDA Streams**: Multiple streams for overlapped GPU work
3. **In-Order Output**: Event-driven completion tracking ensures correct output order
4. **Memory Pools**: Efficient buffer management to minimize allocations
5. **Autotuning**: Analyzes GPU memory bandwidth and PCIe capabilities

The tool is designed to saturate both GPU compute and PCIe bandwidth while maintaining streaming behavior suitable for large datasets.

Use Cases and Integration
-------------------------

### What nvcz Can Be Used For

nvcz excels at compressing large datasets where:
- **GPU resources are available** and you want to maximize compression throughput
- **Storage or network bandwidth** is a bottleneck and compression can help
- **Real-time or streaming** data needs to be compressed efficiently
- **Batch processing** of large files where speed matters more than maximum compression ratio

**Performance Benefits:**
- **10-50x faster** than CPU-based compressors for large files
- **Multi-GPU scaling** allows linear performance improvements with additional GPUs
- **Overlapped I/O** prevents compression from becoming an I/O bottleneck

### Integration with Existing Pipelines

nvcz can be integrated into many existing data processing workflows:

#### Data Lake and ETL Pipelines
```bash
# Compress raw data before storing in data lake
cat sensor_data.json | nvcz compress --algo lz4 --auto > /data-lake/sensors/data.nvcz

# Decompress during ETL processing
nvcz decompress < /data-lake/sensors/data.nvcz | process_etl.py
```

#### Log Aggregation Systems
```bash
# Compress logs in real-time
tail -f /var/log/application.log | nvcz compress --algo lz4 --chunk-mb 8 > logs.nvcz

# Batch compress historical logs
find /var/log/archive -name "*.log" -exec cat {} \; | nvcz compress --algo gdeflate --mgpu > archive.nvcz
```

#### Scientific Data Processing
```bash
# Compress large simulation outputs
cat simulation_results.bin | nvcz compress --algo gdeflate --auto --mgpu > results.nvcz

# Compress genomic data
cat genome.fasta | nvcz compress --algo zstd --auto > genome.nvcz
```

#### Database Backup Compression
```bash
# Stream database dumps through nvcz
pg_dump mydatabase | nvcz compress --algo gdeflate --auto > backup.nvcz
mysqldump mydatabase | nvcz compress --algo lz4 --chunk-mb 64 > backup.nvcz
```

#### Network Data Transfer
```bash
# Compress data before sending over network
cat large_file.bin | nvcz compress --algo lz4 --auto | nc server.example.com 9999

# Decompress received data
nc -l 9999 | nvcz decompress > received_file.bin
```

### Pipeline Integration Examples

#### Apache Spark Integration
```bash
# Compress Spark output partitions
spark_job.py | nvcz compress --algo gdeflate --mgpu > output.nvcz
```

#### Docker Integration
```bash
# In Dockerfile - install nvcz
RUN make -C nvcz && make -C nvcz install

# In container - compress application data
cat /app/data/output.dat | nvcz compress --algo lz4 --auto > /backup/data.nvcz
```

#### CI/CD Pipeline Integration
```bash
# Compress build artifacts
tar cf - build_artifacts/ | nvcz compress --algo gdeflate --auto > artifacts.nvcz

# Upload compressed artifacts
curl -X PUT -T artifacts.nvcz https://storage.example.com/artifacts/
```

#### Kafka Stream Processing
```bash
# Consumer side - decompress incoming streams
kafka-console-consumer --bootstrap-server localhost:9092 --topic compressed_data \
  | nvcz decompress | process_data.py

# Producer side - compress outgoing streams
cat data.json | nvcz compress --algo lz4 --chunk-mb 16 \
  | kafka-console-producer --broker-list localhost:9092 --topic compressed_data
```

### Real-World Performance Examples

#### Large Dataset Compression
```bash
# 100GB dataset compression comparison
time cat dataset.bin | gzip -c > dataset.gz        # ~10 minutes
time cat dataset.bin | nvcz compress --algo gdeflate --auto --mgpu > dataset.nvcz  # ~2 minutes
```

#### Log File Archival
```bash
# Archive 1TB of logs
find /var/log -name "*.log" -exec cat {} \; | nvcz compress --algo gdeflate --auto --mgpu > logs_$(date +%Y%m%d).nvcz
```

#### Database Backup
```bash
# Compress PostgreSQL dump
time pg_dump large_database | nvcz compress --algo gdeflate --auto > backup.nvcz
# Results: 50GB database → 15GB compressed in ~3 minutes
```

Examples
--------

Compress a large file with optimal settings:
```bash
cat large_dataset.bin | nvcz compress --algo gdeflate --auto --mgpu > compressed.nvcz
```

Decompress and verify:
```bash
nvcz decompress < compressed.nvcz > restored.bin
```

Use with pipes for streaming:
```bash
# Compress log files as they arrive
tail -f /var/log/system.log | nvcz compress --algo lz4 --chunk-mb 16 > logs.nvcz
```

Performance Tips
----------------

1. Use `--auto` for optimal settings based on your hardware
2. Enable `--mgpu` for systems with multiple GPUs
3. For large files, use `--auto-size` to increase parallelism
4. LZ4 typically offers the best compression speed
5. GDeflate usually provides the best compression ratio
6. Adjust `--chunk-mb` based on available GPU memory

Troubleshooting
---------------

- Ensure CUDA and nvCOMP are properly installed
- Check GPU memory availability for large chunk sizes
- Use `nvidia-smi` to verify GPU status
- For multi-GPU, ensure `CUDA_VISIBLE_DEVICES` is set appropriately

License
-------

This project is provided as-is for educational and research purposes.
