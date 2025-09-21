// src/codec_snappy.cpp
#include "nvcz/codec.hpp"
#include "nvcz/util.hpp"

#include <memory>
#include <unordered_map>

#include <nvcomp/snappy.hpp>                // nvcomp::SnappyManager
#include <nvcomp/nvcompManagerFactory.hpp>  // nvcomp::create_manager
#include <nvcomp.hpp>                       // CompressionConfig / DecompressionConfig, BitstreamKind, ChecksumPolicy

namespace nvcz {

struct NvcompSnappy : Codec {
  const size_t chunk_size;
  const bool enable_checksum;
  std::unordered_map<cudaStream_t, std::shared_ptr<nvcomp::nvcompManagerBase>> mgr_by_stream;

  nvcomp::SnappyManager& get_manager(cudaStream_t s) {
    auto it = mgr_by_stream.find(s);
    if (it != mgr_by_stream.end()) return static_cast<nvcomp::SnappyManager&>(*it->second);
    nvcompBatchedSnappyOpts_t opts{};
    nvcomp::ChecksumPolicy checksum_policy = enable_checksum ?
        nvcomp::ChecksumPolicy::ComputeAndVerify :
        nvcomp::ChecksumPolicy::NoComputeNoVerify;
    auto mgr = std::make_shared<nvcomp::SnappyManager>(
        chunk_size, opts, s, checksum_policy, nvcomp::BitstreamKind::NVCOMP_NATIVE);
    mgr_by_stream.emplace(s, mgr);
    return static_cast<nvcomp::SnappyManager&>(*mgr);
  }

  NvcompSnappy(size_t chunk_size_kb, bool enable_checksum)
    : chunk_size(chunk_size_kb * 1024), enable_checksum(enable_checksum) {}
  const char* name() const override { return enable_checksum ? "nvcomp-snappy-checksum" : "nvcomp-snappy"; }

  // Ask the manager for a safe max compressed buffer size for an input of length n.
  size_t max_compressed_bound(size_t n) const override {
    nvcompBatchedSnappyOpts_t opts{};   // defaults

    nvcomp::ChecksumPolicy checksum_policy = enable_checksum ?
        nvcomp::ChecksumPolicy::ComputeAndVerify :
        nvcomp::ChecksumPolicy::NoComputeNoVerify;

    nvcomp::SnappyManager mgr{
        chunk_size,
        opts,
        /*stream*/ 0,
        checksum_policy,
        nvcomp::BitstreamKind::NVCOMP_NATIVE};

    nvcomp::CompressionConfig cfg = mgr.configure_compression(n);
    return cfg.max_compressed_buffer_size;
  }

  // New contract:
  // - Writes exact compressed size to device pointer d_comp_size.
  // - Copies the *max-bound* payload back to host (dst). The caller will later
  //   write only the first 'comp_len' bytes after reading a pinned size_t
  //   (post-event) that was D2H-copied separately.
  // - No stream syncs here.
  void compress_with_stream(const uint8_t* src, size_t n,
                            uint8_t* dst, cudaStream_t s,
                            size_t* d_comp_size) override
  {
    nvcompBatchedSnappyOpts_t opts{};

    nvcomp::ChecksumPolicy checksum_policy = enable_checksum ?
        nvcomp::ChecksumPolicy::ComputeAndVerify :
        nvcomp::ChecksumPolicy::NoComputeNoVerify;

    auto& mgr = get_manager(s);
    nvcomp::CompressionConfig cfg = mgr.configure_compression(n);

    // Device staging buffers
    void* d_in  = nullptr;
    void* d_out = nullptr;
    cuda_ck(cudaMallocAsync(&d_in,  n, s), "snappy d_in");
    cuda_ck(cudaMallocAsync(&d_out, cfg.max_compressed_buffer_size, s), "snappy d_out");

    // H2D input
    cuda_ck(cudaMemcpyAsync(d_in, src, n, cudaMemcpyHostToDevice, s), "snappy H2D");

    // Compress (nvCOMP writes exact size to d_comp_size on device)
    get_manager(s).compress(
        static_cast<const uint8_t*>(d_in),
        static_cast<uint8_t*>(d_out),
        cfg,
        d_comp_size);

    // Copy the max-bound output payload back to host. Writer will emit only the
    // first 'comp_len' bytes after it reads the pinned size (post-event).
    cuda_ck(cudaMemcpyAsync(
        dst,
        d_out,
        cfg.max_compressed_buffer_size,
        cudaMemcpyDeviceToHost,
        s), "snappy D2H (max-bound)");

    // Return buffers to the async allocator when the stream reaches these ops.
    cuda_ck(cudaFreeAsync(d_in,  s), "snappy free d_in");
    cuda_ck(cudaFreeAsync(d_out, s), "snappy free d_out");
  }

  // Decompress on the provided CUDA stream (no syncs here).
  void decompress_with_stream(const uint8_t* comp, size_t comp_n,
                              uint8_t* dst, size_t raw_n,
                              cudaStream_t s) override
  {
    // Copy compressed data to device
    void* d_in = nullptr;
    cuda_ck(cudaMallocAsync(&d_in, comp_n, s), "snappy d_in (comp)");
    cuda_ck(cudaMemcpyAsync(d_in, comp, comp_n, cudaMemcpyHostToDevice, s), "snappy H2D comp");

    // Create manager from header in device buffer
    std::shared_ptr<nvcomp::nvcompManagerBase> mgr =
        nvcomp::create_manager(static_cast<const uint8_t*>(d_in), s);

    // Configure decompression (comp_size not required for NVCOMP_NATIVE)
    nvcomp::DecompressionConfig dcfg =
        mgr->configure_decompression(static_cast<const uint8_t*>(d_in), /*comp_size*/ nullptr);

    // Device output
    void* d_out = nullptr;
    cuda_ck(cudaMallocAsync(&d_out, raw_n, s), "snappy d_out (raw)");

    // Decompress
    mgr->decompress(
        static_cast<uint8_t*>(d_out),
        static_cast<const uint8_t*>(d_in),
        dcfg,
        /*comp_size*/ nullptr);

    // D2H payload
    cuda_ck(cudaMemcpyAsync(dst, d_out, raw_n, cudaMemcpyDeviceToHost, s), "snappy D2H raw");

    // Free when the stream reaches these ops
    cuda_ck(cudaFreeAsync(d_in,  s), "snappy free d_in");
    cuda_ck(cudaFreeAsync(d_out, s), "snappy free d_out");
  }

  void compress_dd(const void* d_src, size_t n,
                   void* d_dst, cudaStream_t s,
                   size_t* d_comp_size) override
  {
    nvcompBatchedSnappyOpts_t opts{};

    nvcomp::ChecksumPolicy checksum_policy = enable_checksum ?
        nvcomp::ChecksumPolicy::ComputeAndVerify :
        nvcomp::ChecksumPolicy::NoComputeNoVerify;

    nvcomp::SnappyManager mgr{
        chunk_size,
        opts,
        s,
        checksum_policy,
        nvcomp::BitstreamKind::NVCOMP_NATIVE};

    nvcomp::CompressionConfig cfg = mgr.configure_compression(n);

    get_manager(s).compress(
        static_cast<const uint8_t*>(d_src),
        static_cast<uint8_t*>(d_dst),
        cfg,
        d_comp_size);
  }

  void decompress_dd(const void* d_comp, size_t comp_n,
                     void* d_dst, size_t raw_n,
                     cudaStream_t s) override
  {
    (void)comp_n;
    std::shared_ptr<nvcomp::nvcompManagerBase> mgr =
        nvcomp::create_manager(static_cast<const uint8_t*>(d_comp), s);
    nvcomp::DecompressionConfig dcfg =
        mgr->configure_decompression(static_cast<const uint8_t*>(d_comp), /*comp_size*/ nullptr);
    mgr->decompress(static_cast<uint8_t*>(d_dst),
                    static_cast<const uint8_t*>(d_comp),
                    dcfg,
                    /*comp_size*/ nullptr);
  }
};

std::unique_ptr<Codec> make_codec_snappy(size_t chunk_size_kb, bool enable_checksum) { return std::make_unique<NvcompSnappy>(chunk_size_kb, enable_checksum); }

} // namespace nvcz
