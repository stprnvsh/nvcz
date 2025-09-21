// src/main.cpp
#include "nvcz/codec.hpp"
#include "nvcz/framing.hpp"
#include "nvcz/util.hpp"
#include "nvcz/autotune.hpp"
#include "nvcz/mgpu.hpp"   // CLI still supports --mgpu

#include <vector>
#include <string>
#include <cstdio>
#include <memory>
#include <algorithm>
#include <deque>
#include <cstring>
#include <chrono>
#include <iostream>
#include <fstream>

#include "nvcz/nvcz.hpp"  // Include the library API for RingBufferManager

using namespace nvcz;

static void usage() {
  std::fprintf(stderr,
    "nvcz: stream compressor using nvCOMP (LZ4, GDeflate, Snappy, Zstd)\n"
    "Usage:\n"
    "  nvcz compress [input] [output] --algo {lz4|gdeflate|snappy|zstd} [--chunk-mb N] [--nvcomp-chunk-kb N] [--auto] [--streams N]\n"
    "                [--mgpu] [--gpus all|0,2,3] [--streams-per-gpu N] [--auto-size] [--progress]\n"
    "  nvcz decompress [input] [output] [--auto] [--streams N]\n"
    "                   [--mgpu] [--gpus all|0,2,3] [--streams-per-gpu N] [--auto-size] [--progress]\n"
    "Options:\n"
    "  --progress    Show progress bar for long operations\n"
    "  --input FILE  Read from FILE instead of stdin\n"
    "  --output FILE Write to FILE instead of stdout\n"
    "Examples:\n"
    "  cat in.bin | nvcz compress --algo lz4 --auto > out.nvcz\n"
    "  nvcz decompress < out.nvcz > out.bin\n"
    "  nvcz compress input.bin output.nvcz --algo gdeflate --auto --progress\n"
    "  nvcz compress --input data.bin --output compressed.nvcz --mgpu\n"
    "  cat big.bin | nvcz compress --algo gdeflate --nvcomp-chunk-kb 256 --mgpu --gpus 0,1 > big.nvcz\n");
}

static Algo parse_algo(const std::string& s) {
  if (s == "lz4")      return Algo::LZ4;
  if (s == "gdeflate") return Algo::GDEFLATE;
  if (s == "snappy")   return Algo::SNAPPY;
  if (s == "zstd")     return Algo::ZSTD;
  die("unknown --algo");
  return Algo::LZ4;
}

// Simple progress callback for CLI
static void progress_callback(size_t processed, size_t total) {
  if (total == 0) return;

  int percent = static_cast<int>((processed * 100) / total);
  int bar_width = 40;
  int filled = (percent * bar_width) / 100;

  std::fprintf(stderr, "\rProgress: %3d%% [", percent);
  for (int i = 0; i < filled; i++) std::fprintf(stderr, "█");
  for (int i = filled; i < bar_width; i++) std::fprintf(stderr, " ");
  std::fprintf(stderr, "] %zu/%zu bytes", processed, total);

  if (processed == total) {
    std::fprintf(stderr, "\n");
  }
  std::fflush(stderr);
}

// File I/O helpers
static std::unique_ptr<std::ifstream> open_input_file(const std::string& filename) {
  auto file = std::make_unique<std::ifstream>(filename, std::ios::binary);
  if (!file->is_open()) {
    std::fprintf(stderr, "Error: Cannot open input file '%s'\n", filename.c_str());
    return nullptr;
  }
  return file;
}

static std::unique_ptr<std::ofstream> open_output_file(const std::string& filename) {
  auto file = std::make_unique<std::ofstream>(filename, std::ios::binary);
  if (!file->is_open()) {
    std::fprintf(stderr, "Error: Cannot open output file '%s'\n", filename.c_str());
    return nullptr;
  }
  return file;
}

// ---------------- pinned + overlapped single-GPU compress ----------------
//
// New pattern:
//  - Codec::compress_with_stream(src,n,dst,stream,d_comp_size)
//  - Worker/SG path issues cudaMemcpyAsync(host_size <- d_comp_size) and fences with an event.
//  - Writer reads *(host_size) after event and writes exactly that many bytes.

static void cmd_compress(Algo algo, uint32_t chunk_mb, bool auto_tune, int cli_streams, size_t nvcomp_chunk_kb)
{
  AutoTune t{};
  if (auto_tune) t = pick_tuning(/*verbose=*/true);
  if (!auto_tune) t.chunk_mb = chunk_mb;
  const uint32_t CHUNK_MB = t.chunk_mb;
  const int streams = cli_streams > 0 ? cli_streams : (auto_tune ? t.streams : 3);

  auto codec = make_codec(algo, nvcomp_chunk_kb);
  if (!codec) die("codec not available");

  Header h{}; std::memcpy(h.magic, MAGIC, 5);
  h.version = 1; h.algo = (uint8_t)algo; h.chunk_mb = CHUNK_MB;
  nvcz::fwrite_exact(&h, sizeof(h), stdout);

  const size_t CHUNK = size_t(CHUNK_MB) * 1024 * 1024;

  // CUDA streams
  std::vector<cudaStream_t> ss(streams);
  for (int i=0;i<streams;++i) cuda_ck(cudaStreamCreate(&ss[i]), "mk stream");

  // Enhanced ring buffer system (same as library uses)
  const int RING_SLOTS = std::max(3, streams * 2); // More sophisticated than basic ping-pong
  RingBufferConfig ring_config;
  ring_config.buffer_size_mb = CHUNK_MB;
  ring_config.ring_slots = static_cast<size_t>(RING_SLOTS);
  ring_config.enable_overlapped_io = true;

  RingBufferManager ring_buffer(ring_config);
  if (!ring_buffer.initialize(static_cast<size_t>(streams))) {
    die("Failed to initialize ring buffer manager");
  }

  // Per-stream device-side size tracking (reused across ring buffer slots)
  std::vector<size_t*> d_comp_sizes(streams, nullptr);
  std::vector<size_t*> h_comp_sizes(streams);
  for (int i=0;i<streams;++i) {
    cuda_ck(cudaHostAlloc(&h_comp_sizes[i], sizeof(size_t), cudaHostAllocDefault), "host comp size");
    cuda_ck(cudaMallocAsync(reinterpret_cast<void**>(&d_comp_sizes[i]), sizeof(size_t), ss[i]), "malloc d_comp_size");
  }

  // allocate buffers to worst-case bound once
  const size_t worst = codec->max_compressed_bound(CHUNK);

  struct Job {
    uint64_t     seq = 0;
    int          stream = 0;
    size_t       raw_len = 0;
    cudaEvent_t  done = nullptr;
    size_t*      h_comp_size = nullptr; // pointer to host-side compressed size buffer
    size_t*      d_comp_size = nullptr; // pointer to device-side compressed size buffer
  };

  std::deque<Job> inflight; inflight.clear();
  uint64_t next_seq = 0, write_seq = 0;

  auto try_flush = [&](){
    while (!inflight.empty() && inflight.front().seq == write_seq) {
      Job &j = inflight.front();
      cudaError_t q = cudaEventQuery(j.done);
      if (q == cudaErrorNotReady) break;
      cuda_ck(q, "event query (c)");
      cuda_ck(cudaEventSynchronize(j.done), "event sync (c)");
      cuda_ck(cudaEventDestroy(j.done), "event destroy (c)");

      // exact compressed size is now safely in the pinned host size_t
      size_t comp_len = *(j.h_comp_size);
      uint64_t r = j.raw_len;

      // For ring buffer implementation, we need to get the output buffer
      // that corresponds to this job (simplified for now)
      auto [output_buffer, output_size] = ring_buffer.get_output_buffer();
      nvcz::fwrite_exact(&r, sizeof(r), stdout);
      nvcz::fwrite_exact(&comp_len, sizeof(comp_len), stdout);
      nvcz::fwrite_exact(output_buffer, comp_len, stdout);

      write_seq++;
      inflight.pop_front();
    }
  };

  bool eof = false;
  size_t stream_index = 0;

  while (!eof) {
    for (int i=0;i<streams; ++i) {
      // Get input buffer from ring buffer manager
      auto [input_buffer, input_buffer_size] = ring_buffer.get_input_buffer();

      // read next chunk into ring buffer
      size_t got = read_chunk_into_ptr(input_buffer, CHUNK);
      if (got == 0) { eof = true; continue; }

      // Mark input buffer as filled
      ring_buffer.mark_input_buffer_filled(input_buffer, got);

      // Get output buffer from ring buffer manager
      auto [output_buffer, output_buffer_size] = ring_buffer.get_output_buffer();

      // Get device and host size buffers for this stream
      size_t* d_comp_size = d_comp_sizes[i];
      size_t* h_comp_size = h_comp_sizes[i];

      // launch async compress on stream i; codec copies up to bound to output_buffer, writes true size to dev size
      codec->compress_with_stream(
          input_buffer, got,
          output_buffer,
          ss[i],
          d_comp_size);

      // also stage the size to host pinned
      cuda_ck(cudaMemcpyAsync(h_comp_size, d_comp_size, sizeof(size_t),
                              cudaMemcpyDeviceToHost, ss[i]), "D2H comp size (ring)");

      // fence completion with an event
      cudaEvent_t ev; cuda_ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "mk event (ring)");
      cuda_ck(cudaEventRecord(ev, ss[i]), "record event (ring)");

      Job j;
      j.seq = next_seq++;
      j.stream = i;
      j.raw_len = got;
      j.done = ev;
      j.h_comp_size = h_comp_size;
      j.d_comp_size = d_comp_size;
      inflight.push_back(j);

      stream_index = (stream_index + 1) % streams;
    }

    try_flush();
  }

  // drain remaining jobs in order
  while (!inflight.empty()) {
    Job &j = inflight.front();
    cuda_ck(cudaEventSynchronize(j.done), "drain event (c)");
    cuda_ck(cudaEventDestroy(j.done), "destroy event (c)");

    size_t comp_len = *(j.h_comp_size);
    uint64_t r = j.raw_len;

    // Get output buffer for this job (simplified)
    auto [output_buffer, output_size] = ring_buffer.get_output_buffer();
    nvcz::fwrite_exact(&r, sizeof(r), stdout);
    nvcz::fwrite_exact(&comp_len, sizeof(comp_len), stdout);
    nvcz::fwrite_exact(output_buffer, comp_len, stdout);

    write_seq++;
    inflight.pop_front();
  }

  // trailer
  uint64_t z=0; nvcz::fwrite_exact(&z,8,stdout); nvcz::fwrite_exact(&z,8,stdout);

  // Cleanup host and device size buffers
  for (int i=0;i<streams;++i){
    if (h_comp_sizes[i]) cudaFreeHost(h_comp_sizes[i]);
    if (d_comp_sizes[i]) cudaFreeAsync(d_comp_sizes[i], ss[i]);
  }

  // Ring buffer manager will clean up automatically
  for (auto s : ss) cuda_ck(cudaStreamDestroy(s), "rm stream");
}

// ---------------- pinned + overlapped single-GPU decompress ----------------

static void cmd_decompress(bool auto_tune, int cli_streams, size_t nvcomp_chunk_kb)
{
  Header h{}; nvcz::fread_exact(&h, sizeof(h), stdin);
  if (std::memcmp(h.magic, MAGIC, 5)!=0 || h.version!=1) die("bad header");
  auto algo = (Algo)h.algo;

  AutoTune t{}; if (auto_tune) t = pick_tuning(true);
  const uint32_t CHUNK_MB = h.chunk_mb; // honor file’s chunk size
  const int streams = cli_streams > 0 ? cli_streams : (auto_tune ? t.streams : 3);

  auto codec = make_codec(algo, nvcomp_chunk_kb);
  if (!codec) die("codec not available");

  const size_t CHUNK = size_t(CHUNK_MB) * 1024 * 1024;

  std::vector<cudaStream_t> ss(streams);
  for (int i=0;i<streams;++i) cuda_ck(cudaStreamCreate(&ss[i]), "mk d stream");

  // Enhanced ring buffer system for decompression
  const int RING_SLOTS = std::max(3, streams * 2);
  RingBufferConfig ring_config;
  ring_config.buffer_size_mb = CHUNK_MB;
  ring_config.ring_slots = static_cast<size_t>(RING_SLOTS);
  ring_config.enable_overlapped_io = true;

  RingBufferManager ring_buffer(ring_config);
  if (!ring_buffer.initialize(static_cast<size_t>(streams))) {
    die("Failed to initialize ring buffer manager for decompression");
  }

  struct Job {
    uint64_t     seq = 0;
    int          stream = 0;
    size_t       raw_len = 0;
    cudaEvent_t  done = nullptr;
  };

  std::deque<Job> inflight; inflight.clear();
  uint64_t next_seq = 0, write_seq = 0;

  auto try_flush = [&](){
    while (!inflight.empty() && inflight.front().seq == write_seq) {
      Job &j = inflight.front();
      cudaError_t q = cudaEventQuery(j.done);
      if (q == cudaErrorNotReady) break;
      cuda_ck(q, "event query (d)");
      cuda_ck(cudaEventSynchronize(j.done), "event sync (d)");
      cuda_ck(cudaEventDestroy(j.done), "event destroy (d)");

      // Get output buffer from ring buffer manager
      auto [output_buffer, output_size] = ring_buffer.get_output_buffer();
      nvcz::fwrite_exact(output_buffer, j.raw_len, stdout);

      write_seq++;
      inflight.pop_front();
    }
  };

  // read triplets and enqueue across streams in round-robin
  for (;;) {
    bool queued_any = false;
    for (int i=0;i<streams; ++i) {
      // peek header
      uint64_t r=0, c=0;
      size_t got_r = std::fread(&r,1,sizeof(r),stdin);
      size_t got_c = std::fread(&c,1,sizeof(c),stdin);
      if (got_r != sizeof(r) || got_c != sizeof(c)) die("truncated header");
      if (r==0 && c==0) { queued_any = false; goto done_read; }

      // Get input and output buffers from ring buffer manager
      auto [input_buffer, input_buffer_size] = ring_buffer.get_input_buffer();
      auto [output_buffer, output_buffer_size] = ring_buffer.get_output_buffer();

      // Ensure buffers are large enough
      if (input_buffer_size < c) {
        die("Input buffer too small for compressed data");
      }
      if (output_buffer_size < r) {
        die("Output buffer too small for decompressed data");
      }

      nvcz::fread_exact(input_buffer, c, stdin);

      // Mark input buffer as filled
      ring_buffer.mark_input_buffer_filled(input_buffer, c);

      // launch async decomp
      codec->decompress_with_stream(
        input_buffer, c,
        output_buffer, r,
        ss[i]);

      cudaEvent_t ev; cuda_ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "mk event (d)");
      cuda_ck(cudaEventRecord(ev, ss[i]), "record event (d)");

      Job j; j.seq = next_seq++; j.stream = i; j.raw_len = r; j.done = ev;
      inflight.push_back(j);

      queued_any = true;
    }
    try_flush();
    if (!queued_any) break;
  }

done_read:
  // drain remaining
  while (!inflight.empty()) {
    Job &j = inflight.front();
    cuda_ck(cudaEventSynchronize(j.done), "drain event (d)");
    cuda_ck(cudaEventDestroy(j.done), "destroy event (d)");

    // Get output buffer from ring buffer manager
    auto [output_buffer, output_size] = ring_buffer.get_output_buffer();
    nvcz::fwrite_exact(output_buffer, j.raw_len, stdout);

    write_seq++;
    inflight.pop_front();
  }

  for (auto s : ss) cuda_ck(cudaStreamDestroy(s), "rm d stream");
}

// ---------------- multi-GPU aware main (CLI unchanged) ----------------

// Enhanced file-based compression using library API
static int cmd_compress_file(const std::string& input_file, const std::string& output_file,
                           Algo algo, uint32_t chunk_mb, bool auto_tune, bool show_progress,
                           size_t nvcomp_chunk_kb) {
  try {
    Config config;
    config.algorithm = algo;
    config.chunk_mb = chunk_mb;
    config.nvcomp_chunk_kb = nvcomp_chunk_kb;
    config.enable_autotune = auto_tune;

    if (show_progress) {
      std::fprintf(stderr, "Compressing %s -> %s\n", input_file.c_str(), output_file.c_str());
    }

    auto start_time = std::chrono::steady_clock::now();

    auto result = compress_file(input_file, output_file, config,
      show_progress ? progress_callback : nullptr);

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (result) {
      if (show_progress) {
        std::fprintf(stderr, "\nCompression completed in %.2fs\n",
                    duration.count() / 1000.0);
        std::fprintf(stderr, "Input: %zu bytes, Output: %zu bytes, Ratio: %.2fx\n",
                    result.stats.input_bytes, result.stats.compressed_bytes,
                    result.stats.compression_ratio);
      }
      return 0;
    } else {
      std::fprintf(stderr, "Error: %s\n", result.error_message.c_str());
      return 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Compression failed: %s\n", e.what());
    return 1;
  }
}

// Enhanced file-based decompression using library API
static int cmd_decompress_file(const std::string& input_file, const std::string& output_file,
                              bool auto_tune, bool show_progress) {
  try {
    Config config;
    config.enable_autotune = auto_tune;

    if (show_progress) {
      std::fprintf(stderr, "Decompressing %s -> %s\n", input_file.c_str(), output_file.c_str());
    }

    auto start_time = std::chrono::steady_clock::now();

    auto result = decompress_file(input_file, output_file, config,
      show_progress ? progress_callback : nullptr);

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (result) {
      if (show_progress) {
        std::fprintf(stderr, "\nDecompression completed in %.2fs\n",
                    duration.count() / 1000.0);
        std::fprintf(stderr, "Input: %zu bytes, Output: %zu bytes, Ratio: %.2fx\n",
                    result.stats.input_bytes, result.stats.compressed_bytes,
                    result.stats.compression_ratio);
      }
      return 0;
    } else {
      std::fprintf(stderr, "Error: %s\n", result.error_message.c_str());
      return 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Decompression failed: %s\n", e.what());
    return 1;
  }
}

int main(int argc, char** argv)
{
  if (argc < 2) { usage(); return 1; }
  std::string mode = argv[1];

  // Check for file-based operations first
  if (mode == "compress" || mode == "decompress") {
    std::string input_file, output_file;
    bool show_progress = false;
    std::string input_option, output_option;

    // Parse arguments
    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--input" && i + 1 < argc) {
        input_option = argv[++i];
      } else if (arg == "--output" && i + 1 < argc) {
        output_option = argv[++i];
      } else if (arg == "--progress") {
        show_progress = true;
      } else if (arg == "--help" || arg == "-h") {
        usage();
        return 0;
      } else if (input_file.empty() && arg[0] != '-') {
        input_file = arg;
      } else if (output_file.empty() && arg[0] != '-') {
        output_file = arg;
      }
    }

    // Determine actual input/output sources
    std::string actual_input = input_option.empty() ? input_file : input_option;
    std::string actual_output = output_option.empty() ? output_file : output_option;

    // If we have file arguments, use library-based file operations
    if (!actual_input.empty() && !actual_output.empty()) {
      if (mode == "compress") {
        std::string algo_str = "lz4";
        uint32_t chunk_mb = 32;
        bool auto_tune = false;
        size_t nvcomp_chunk_kb = 64;

        // Parse remaining options
        for (int i = 2; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--algo" && i + 1 < argc) {
            algo_str = argv[++i];
          } else if (arg == "--chunk-mb" && i + 1 < argc) {
            chunk_mb = std::max(1, std::atoi(argv[++i]));
          } else if (arg == "--nvcomp-chunk-kb" && i + 1 < argc) {
            nvcomp_chunk_kb = std::max(1UL, (size_t)std::atoi(argv[++i]));
          } else if (arg == "--auto") {
            auto_tune = true;
          }
        }

        return cmd_compress_file(actual_input, actual_output,
                               parse_algo(algo_str), chunk_mb, auto_tune, show_progress, nvcomp_chunk_kb);
      } else {
        bool auto_tune = false;

        // Parse remaining options
        for (int i = 2; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--auto") {
            auto_tune = true;
          }
        }

        return cmd_decompress_file(actual_input, actual_output, auto_tune, show_progress);
      }
    }
  }

  // Fall back to original streaming mode
  std::string algo_str = "lz4";
  uint32_t chunk_mb = 32;
  bool auto_tune=false; int streams=0;
  size_t nvcomp_chunk_kb = 64;  // Default 64KB nvCOMP chunks

  // MGPU flags
  bool mgpu=false, auto_size=false;
  int streams_per_gpu_override=0;
  std::vector<int> gpu_ids_override;

  for (int i=2;i<argc;++i) {
    std::string a = argv[i];
    if (a=="--algo" && i+1<argc) { algo_str = argv[++i]; }
    else if (a=="--chunk-mb" && i+1<argc) { chunk_mb = std::max(1, std::atoi(argv[++i])); }
    else if (a=="--nvcomp-chunk-kb" && i+1<argc) { nvcomp_chunk_kb = std::max(1UL, (size_t)std::atoi(argv[++i])); }
    else if (a=="--auto") { auto_tune = true; }
    else if (a=="--streams" && i+1<argc) { streams = std::max(1, std::atoi(argv[++i])); }
    else if (a=="--mgpu") { mgpu = true; }
    else if (a=="--auto-size") { auto_size = true; }
    else if (a=="--streams-per-gpu" && i+1<argc) { streams_per_gpu_override = std::max(1, std::atoi(argv[++i])); }
    else if (a=="--gpus" && i+1<argc) {
      std::string list = argv[++i];
      if (list != "all") {
        size_t pos=0;
        while (pos < list.size()) {
          size_t comma = list.find(',', pos);
          int id = std::atoi(list.substr(pos, (comma==std::string::npos? list.size():comma)-pos).c_str());
          gpu_ids_override.push_back(id);
          if (comma==std::string::npos) break;
          pos = comma+1;
        }
      }
    } else { usage(); return 1; }
  }

  try {
    if (mode=="compress") {
      if (mgpu) {
        AutoTune base = auto_tune ? pick_tuning(/*verbose=*/true)
                                  : AutoTune{chunk_mb, std::max(1, streams)};
        auto t = pick_mgpu_tuning(base, /*size_aware=*/auto_size,
                                  streams_per_gpu_override, gpu_ids_override);
        if (!auto_tune) t.chunk_mb = chunk_mb;

        // Use file-based MGPU if we have file arguments, otherwise use stdin/stdout
        if (!actual_input.empty() && !actual_output.empty()) {
          // For file operations, we need to open files and use the enhanced MGPU function
          auto input_file = open_input_file(actual_input.c_str());
          auto output_file = open_output_file(actual_output.c_str());
          if (!input_file || !output_file) {
            return 1;
          }

          compress_mgpu_with_files(parse_algo(algo_str), t,
                                 input_file.get(), output_file.get(),
                                 show_progress ? progress_callback : nullptr);
        } else {
          compress_mgpu(parse_algo(algo_str), t);
        }
      } else {
        cmd_compress(parse_algo(algo_str), chunk_mb, auto_tune, streams, nvcomp_chunk_kb);
      }
    } else if (mode=="decompress") {
      if (mgpu) {
        AutoTune base = auto_tune ? pick_tuning(/*verbose=*/true)
                                  : AutoTune{chunk_mb, std::max(1, streams)};
        auto t = pick_mgpu_tuning(base, /*size_aware=*/auto_size,
                                  streams_per_gpu_override, gpu_ids_override);

        // Use file-based MGPU if we have file arguments, otherwise use stdin/stdout
        if (!actual_input.empty() && !actual_output.empty()) {
          // For file operations, we need to open files and use the enhanced MGPU function
          auto input_file = open_input_file(actual_input.c_str());
          auto output_file = open_output_file(actual_output.c_str());
          if (!input_file || !output_file) {
            return 1;
          }

          decompress_mgpu_with_files(t, input_file.get(), output_file.get(),
                                    show_progress ? progress_callback : nullptr);
        } else {
          decompress_mgpu(t);
        }
      } else {
        cmd_decompress(auto_tune, streams, nvcomp_chunk_kb);
      }
    } else {
      usage(); return 1;
    }
  } catch (...) {
    std::fprintf(stderr, "fatal: unhandled exception\n");
    return 2;
  }
  return 0;
}
