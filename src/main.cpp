// src/main.cpp
#include "nvcz/codec.hpp"
#include "nvcz/framing.hpp"
#include "nvcz/util.hpp"
#include "nvcz/autotune.hpp"
#include "nvcz/mgpu.hpp"   // CLI still supports --mgpu
#include "nvcz/gds.hpp"

#include <vector>
#include <string>
#include <cstdio>
#include <memory>
#include <algorithm>
#include <deque>
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>
#include <fcntl.h>

using namespace nvcz;

static void usage() {
  std::fprintf(stderr,
    "nvcz: stream compressor using nvCOMP (LZ4, GDeflate, Snappy, Zstd)\n"
    "Usage:\n"
    "  nvcz compress --algo {lz4|gdeflate|snappy|zstd} [--chunk-mb N] [--nvcomp-chunk-kb N] [--auto] [--streams N]\n"
    "                [--mgpu] [--gpus all|0,2,3] [--streams-per-gpu N] [--auto-size] [--progress] [--checksum]\n"
    "                [-i input_file] [-o output_file] [input_files...] [--gds]\n"
    "  nvcz decompress [--auto] [--streams N] [--progress] [--checksum]\n"
    "                  [--mgpu] [--gpus all|0,2,3] [--streams-per-gpu N] [--auto-size]\n"
    "                  [-i input_file] [-o output_file] [input_files...] [--gds]\n"
    "Examples:\n"
    "  cat in.bin | nvcz compress --algo lz4 --auto --progress --checksum > out.nvcz\n"
    "  nvcz decompress --progress --checksum < out.nvcz > out.bin\n"
    "  nvcz compress --algo lz4 -i input.bin -o output.nvcz --progress --checksum\n"
    "  nvcz decompress -i input.nvcz -o output.bin --progress --checksum\n"
    "  nvcz compress --algo gdeflate --nvcomp-chunk-kb 256 --mgpu --gpus 0,1 -i big.bin -o big.nvcz --progress --checksum\n");
}
static void cmd_compress_gds(Algo algo, uint32_t chunk_mb, size_t nvcomp_chunk_kb,
                             bool show_progress, bool enable_checksum, const std::string& in_path,
                             const std::string& out_path)
{
  // Progress tracking
  std::atomic<uint64_t> input_bytes{0};
  std::atomic<uint64_t> output_bytes{0};
  std::atomic<uint64_t> chunks_processed{0};

  // Open files and GDS handles
  int in_fd  = ::open(in_path.c_str(),  O_RDONLY);
  int out_fd = ::open(out_path.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0644);
  if (in_fd < 0 || out_fd < 0) die("open failed (gds)");

  nvcz::GDSFile gds_in;  if (!gds_in.open_fd(in_fd))  die("GDS open input failed");
  nvcz::GDSFile gds_out; if (!gds_out.open_fd(out_fd)) die("GDS open output failed");

  auto codec = make_codec(algo, nvcomp_chunk_kb, enable_checksum);
  if (!codec) die("codec not available");

  // Header via pwrite
  Header h{}; std::memcpy(h.magic, MAGIC, 5);
  h.version = 1; h.algo = (uint8_t)algo; h.chunk_mb = chunk_mb;
  off_t out_off = 0;
  if (::pwrite(out_fd, &h, sizeof(h), out_off) != (ssize_t)sizeof(h)) die("pwrite header (gds)");
  out_off += sizeof(h);
  output_bytes.fetch_add(sizeof(h));

  // Progress thread
  std::thread progress_thread; std::atomic<bool> progress_running{true};
  if (show_progress) {
    progress_thread = std::thread([&](){
      auto start_time = std::chrono::steady_clock::now();
      while (progress_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        if (elapsed > 0) {
          double in_mb = input_bytes.load() / (1024.0*1024.0);
          double out_mb = output_bytes.load() / (1024.0*1024.0);
          double ratio = input_bytes.load() ? (double)output_bytes.load() / input_bytes.load() : 0.0;
          double speed_in = in_mb / elapsed;
          double speed_out = out_mb / elapsed;
          std::fprintf(stderr, "\rGDS Progress: %.1f MB in, %.1f MB out (%.2fx), %.1f MB/s in, %.1f MB/s out, %lu chunks",
                       in_mb, out_mb, ratio, speed_in, speed_out, chunks_processed.load());
          std::fflush(stderr);
        }
      }
    });
  }

  const size_t CHUNK = size_t(chunk_mb) * 1024 * 1024;
  const size_t WORST = codec->max_compressed_bound(CHUNK);

  // Device buffers + size ptr (use stream-ordered allocations and mempool)
  cudaStream_t s; cuda_ck(cudaStreamCreate(&s), "gds stream");
  {
    int dev=0; cuda_ck(cudaGetDevice(&dev), "get device");
    cudaMemPool_t pool{}; cuda_ck(cudaDeviceGetDefaultMemPool(&pool, dev), "get mempool");
    unsigned long long thr = ~0ull; cuda_ck(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr), "set mempool threshold");
  }
  void* d_in=nullptr;  void* d_out=nullptr;  size_t* d_sz=nullptr;  void* d_hdr=nullptr;
  cuda_ck(cudaMallocAsync(&d_in, CHUNK, s), "gds d_in");
  cuda_ck(cudaMallocAsync(&d_out, WORST, s), "gds d_out");
  cuda_ck(cudaMallocAsync((void**)&d_sz, sizeof(size_t), s), "gds d_sz");
  cuda_ck(cudaMallocAsync(&d_hdr, 16, s), "gds d_hdr");

  size_t comp_len_host=0;

  off_t in_off = 0;
  for (;;) {
    // Read file → GPU
    ssize_t r = gds_in.read_to_gpu(d_in, CHUNK, in_off);
    if (r < 0) die("GDS read failed");
    if (r == 0) break; // EOF
    in_off += r; input_bytes.fetch_add(r);

    // Compress on GPU
    codec->compress_dd(d_in, (size_t)r, d_out, s, d_sz);

    // Build header fully on device (H2D raw_len, D2D comp_len)
    cuda_ck(cudaMemcpyAsync(d_hdr, &r, sizeof(uint64_t), cudaMemcpyHostToDevice, s), "gds H2D header raw_len");
    cuda_ck(cudaMemcpyAsync(static_cast<uint8_t*>(d_hdr)+8, d_sz, sizeof(size_t), cudaMemcpyDeviceToDevice, s), "gds D2D header comp_len");
    cuda_ck(cudaStreamSynchronize(s), "gds sync");
    // Fetch comp_len to host only to provide length to cuFileWrite (API requires size on host)
    cuda_ck(cudaMemcpy(&comp_len_host, d_sz, sizeof(size_t), cudaMemcpyDeviceToHost), "gds D2H size");

    // Frame header via pwrite
    uint64_t raw_len = (uint64_t)r;
    // Write header from device buffer (zero-copy)
    ssize_t wh = gds_out.write_from_gpu(d_hdr, 16, out_off);
    if (wh != 16) die("GDS write header failed");
    out_off += 16;
    output_bytes.fetch_add(16);

    // Write GPU buffer → file
    ssize_t w = gds_out.write_from_gpu(d_out, comp_len_host, out_off);
    if (w != (ssize_t)comp_len_host) die("GDS write failed");
    out_off += w; output_bytes.fetch_add(w);
    chunks_processed.fetch_add(1);
  }

  // Trailer
  uint64_t z=0; if (::pwrite(out_fd, &z, 8, out_off) != 8) die("pwrite z1"); out_off+=8;
  if (::pwrite(out_fd, &z, 8, out_off) != 8) die("pwrite z2"); out_off+=8; output_bytes.fetch_add(16);

  // Ensure all work is done before tearing down nvCOMP manager and stream
  cuda_ck(cudaStreamSynchronize(s), "gds cleanup sync");
  // Destroy codec (nvCOMP manager) while stream/context are alive
  codec.reset();
  // Free device buffers and destroy stream
  cudaFree(d_in); cudaFree(d_out); cudaFree(d_sz); cudaFree(d_hdr);
  cudaStreamDestroy(s);

  if (show_progress) {
    progress_running = false; progress_thread.join();
    double in_mb = input_bytes.load() / (1024.0*1024.0);
    double out_mb = output_bytes.load() / (1024.0*1024.0);
    double ratio = input_bytes.load() ? (double)output_bytes.load() / input_bytes.load() : 0.0;
    std::fprintf(stderr, "\nGDS Final: %.1f MB in, %.1f MB out (%.2fx), %lu chunks\n",
                 in_mb, out_mb, ratio, chunks_processed.load());
  }
}

static void cmd_decompress_gds(int cli_streams, size_t nvcomp_chunk_kb,
                               bool show_progress, const std::string& in_path,
                               const std::string& out_path)
{
  std::atomic<uint64_t> input_bytes{0};
  std::atomic<uint64_t> output_bytes{0};
  std::atomic<uint64_t> chunks_processed{0};

  int in_fd  = ::open(in_path.c_str(),  O_RDONLY);
  int out_fd = ::open(out_path.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0644);
  if (in_fd < 0 || out_fd < 0) die("open failed (gds d)");

  nvcz::GDSFile gds_in;  if (!gds_in.open_fd(in_fd))  die("GDS open input failed");
  nvcz::GDSFile gds_out; if (!gds_out.open_fd(out_fd)) die("GDS open output failed");

  // Read header via pread
  Header h{}; if (::pread(in_fd, &h, sizeof(h), 0) != (ssize_t)sizeof(h)) die("pread header");
  if (std::memcmp(h.magic, MAGIC, 5)!=0 || h.version!=1) die("bad header");
  input_bytes.fetch_add(sizeof(h));
  auto algo = (Algo)h.algo;
  const size_t CHUNK = size_t(h.chunk_mb) * 1024 * 1024;

  auto codec = make_codec(algo, nvcomp_chunk_kb);
  if (!codec) die("codec not available");

  void* d_in=nullptr; void* d_out=nullptr; cuda_ck(cudaMalloc(&d_in, codec->max_compressed_bound(CHUNK)), "gds d_in d");
  cuda_ck(cudaMalloc(&d_out, CHUNK), "gds d_out d");
  cudaStream_t s; cuda_ck(cudaStreamCreate(&s), "gds stream d");

  // Progress thread
  std::thread progress_thread; std::atomic<bool> progress_running{true};
  if (show_progress) {
    progress_thread = std::thread([&](){
      auto start_time = std::chrono::steady_clock::now();
      while (progress_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        if (elapsed > 0) {
          double in_mb = input_bytes.load() / (1024.0*1024.0);
          double out_mb = output_bytes.load() / (1024.0*1024.0);
          double ratio = out_mb > 0 ? in_mb/out_mb : 0.0;
          double speed_in = in_mb / elapsed;
          double speed_out = out_mb / elapsed;
          std::fprintf(stderr, "\rGDS Decomp: %.1f MB in, %.1f MB out (%.2fx), %.1f MB/s in, %.1f MB/s out, %lu chunks",
                       in_mb, out_mb, ratio, speed_in, speed_out, chunks_processed.load());
          std::fflush(stderr);
        }
      }
    });
  }

  off_t in_off = sizeof(h);
  for (;;) {
    uint64_t r=0, c=0;
    if (::pread(in_fd, &r, sizeof(r), in_off) != (ssize_t)sizeof(r)) die("pread r"); in_off += sizeof(r);
    if (::pread(in_fd, &c, sizeof(c), in_off) != (ssize_t)sizeof(c)) die("pread c"); in_off += sizeof(c);
    input_bytes.fetch_add(sizeof(r)+sizeof(c));
    if (r==0 && c==0) break;

    // Read comp payload into GPU
    ssize_t got = gds_in.read_to_gpu(d_in, c, in_off);
    if (got != (ssize_t)c) die("GDS read comp (d)");
    in_off += got; input_bytes.fetch_add(got);

    // Decompress device-to-device
    codec->decompress_dd(d_in, c, d_out, r, s);
    cuda_ck(cudaStreamSynchronize(s), "sync dds");

    // Write raw payload from GPU to file
    ssize_t w = gds_out.write_from_gpu(d_out, r, -1);
    if (w != (ssize_t)r) die("GDS write raw");
    output_bytes.fetch_add(w); chunks_processed.fetch_add(1);
  }

  cudaFree(d_in); cudaFree(d_out); cudaStreamDestroy(s);
  if (show_progress) { progress_running=false; progress_thread.join(); }
}

static Algo parse_algo(const std::string& s) {
  if (s == "lz4")      return Algo::LZ4;
  if (s == "gdeflate") return Algo::GDEFLATE;
  if (s == "snappy")   return Algo::SNAPPY;
  if (s == "zstd")     return Algo::ZSTD;
  die("unknown --algo");
  return Algo::LZ4;
}

// ---------------- pinned + overlapped single-GPU compress ----------------
//
// New pattern:
//  - Codec::compress_with_stream(src,n,dst,stream,d_comp_size)
//  - Worker/SG path issues cudaMemcpyAsync(host_size <- d_comp_size) and fences with an event.
//  - Writer reads *(host_size) after event and writes exactly that many bytes.

static void cmd_compress(Algo algo, uint32_t chunk_mb, bool auto_tune, int cli_streams, size_t nvcomp_chunk_kb, bool show_progress, FILE* input_fp, bool enable_checksum)
{
  AutoTune t{};
  if (auto_tune) t = pick_tuning(/*verbose=*/true);
  if (!auto_tune) t.chunk_mb = chunk_mb;
  const uint32_t CHUNK_MB = t.chunk_mb;
  const int streams = cli_streams > 0 ? cli_streams : (auto_tune ? t.streams : 3);

  auto codec = make_codec(algo, nvcomp_chunk_kb);
  if (!codec) die("codec not available");

  // Progress tracking
  std::atomic<uint64_t> input_bytes{0};
  std::atomic<uint64_t> output_bytes{0};
  std::atomic<uint64_t> chunks_processed{0};

  Header h{}; std::memcpy(h.magic, MAGIC, 5);
  h.version = 1; h.algo = (uint8_t)algo; h.chunk_mb = CHUNK_MB;
  nvcz::fwrite_exact(&h, sizeof(h), stdout);

  // Progress reporter thread
  std::thread progress_thread;
  std::atomic<bool> progress_running{true};

  if (show_progress) {
    progress_thread = std::thread([&input_fp, &progress_running, &input_bytes, &output_bytes, &chunks_processed, &enable_checksum]() {
      auto start_time = std::chrono::steady_clock::now();

      while (progress_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Update every second

        uint64_t in = input_bytes.load();
        uint64_t out = output_bytes.load();
        uint64_t chunks = chunks_processed.load();
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        if (elapsed > 0) {
          double in_mb = in / (1024.0 * 1024.0);
          double out_mb = out / (1024.0 * 1024.0);
          double ratio = in > 0 ? (double)out / in : 0.0;
          double speed_in = in / (1024.0 * 1024.0) / elapsed;
          double speed_out = out / (1024.0 * 1024.0) / elapsed;

          std::fprintf(stderr, "\rProgress: %.1f MB in, %.1f MB out (%.2fx), %.1f MB/s in, %.1f MB/s out, %lu chunks",
                      in_mb, out_mb, ratio, speed_in, speed_out, chunks);
          std::fflush(stderr);
        }
      }
    });
  }

  const size_t CHUNK = size_t(CHUNK_MB) * 1024 * 1024;
  const int    SLOTS = 2; // ping/pong per stream

  // CUDA streams
  std::vector<cudaStream_t> ss(streams);
  for (int i=0;i<streams;++i) cuda_ck(cudaStreamCreate(&ss[i]), "mk stream");

  // Per-stream slots
  struct Slots {
    PinnedBuffer raw[SLOTS];
    PinnedBuffer comp[SLOTS];
    struct Sz {
      size_t* host = nullptr; // pinned host size_t where we async-copy the exact size
      size_t* dev  = nullptr; // device side size_t written by nvCOMP
    } sz[SLOTS];
  };
  std::vector<Slots> host(streams);

  // allocate buffers to worst-case bound once
  const size_t worst = codec->max_compressed_bound(CHUNK);
  for (int i=0;i<streams;++i) {
    for (int s=0;s<SLOTS;++s) {
      host[i].raw[s].alloc(CHUNK);
      // Don't allocate comp buffers here - we'll pre-allocate per stream
      void* tmp=nullptr;
      cuda_ck(cudaHostAlloc(&tmp, sizeof(size_t), cudaHostAllocDefault), "host size");
      host[i].sz[s].host = static_cast<size_t*>(tmp);
    }
    // Pre-allocate device size_t buffers per stream (reused across slots)
    cuda_ck(cudaMallocAsync(reinterpret_cast<void**>(&host[i].sz[0].dev), sizeof(size_t), ss[i]), "malloc d_comp_size (sg)");
  }

  // Pre-allocate compressed buffers per stream (2 per stream for ping-pong, no more dynamic allocation)
  std::vector<std::vector<PinnedBuffer>> comp_buffers(streams);
  for (int i=0;i<streams;++i) {
    comp_buffers[i].resize(SLOTS);
    for (int s=0;s<SLOTS;++s) {
      comp_buffers[i][s].alloc(worst);
    }
  }

  // Persistent device buffers per stream (avoid per-chunk cudaMalloc)
  std::vector<void*> d_in(streams, nullptr);
  std::vector<void*> d_out(streams, nullptr);
  for (int i=0;i<streams;++i) {
    cuda_ck(cudaMalloc(&d_in[i], CHUNK), "malloc d_in (sg)");
    cuda_ck(cudaMalloc(&d_out[i], worst), "malloc d_out (sg)");
  }

  struct Job {
    uint64_t     seq = 0;
    int          stream = 0;
    int          slot = 0; // 0/1 ping-pong
    size_t       raw_len = 0;
    cudaEvent_t  done = nullptr;
  };

  std::vector<int> next_slot(streams, 0);
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
      size_t comp_len = *(host[j.stream].sz[j.slot].host);
      uint64_t r = j.raw_len;
      nvcz::fwrite_exact(&r, sizeof(r), stdout);
      nvcz::fwrite_exact(&comp_len, sizeof(comp_len), stdout);
      nvcz::fwrite_exact(comp_buffers[j.stream][j.slot].bytes(), comp_len, stdout);

      // Update progress counters
      output_bytes.fetch_add(sizeof(r) + sizeof(comp_len) + comp_len);
      chunks_processed.fetch_add(1);

      // device size buffers are pre-allocated per stream and reused

      write_seq++;
      inflight.pop_front();
    }
  };

  bool eof = false;
  while (!eof) {
    for (int i=0;i<streams; ++i) {
      int s = next_slot[i];

      // read next chunk into pinned buffer
      size_t got = read_chunk_into_ptr(host[i].raw[s].bytes(), CHUNK);
      if (got == 0) { eof = true; continue; }

      input_bytes.fetch_add(got);

      // use pre-allocated comp buffer (already sized to worst-case)

      // use pre-allocated device size buffer (reused per stream)

      // H2D input into persistent device buffer
      cuda_ck(cudaMemcpyAsync(d_in[i], host[i].raw[s].bytes(), got,
                              cudaMemcpyHostToDevice, ss[i]), "H2D raw (sg)");

      // launch async device-to-device compress; writes exact size to device size ptr
      codec->compress_dd(
          d_in[i], got,
          d_out[i],
          ss[i],
          host[i].sz[0].dev);

      // stage the size to host pinned
      cuda_ck(cudaMemcpyAsync(host[i].sz[s].host, host[i].sz[0].dev, sizeof(size_t),
                              cudaMemcpyDeviceToHost, ss[i]), "D2H comp size (sg)");

      // D2H bound-sized payload into pinned host buffer (writer trims to true size later)
      cuda_ck(cudaMemcpyAsync(comp_buffers[i][s].bytes(), d_out[i], worst,
                              cudaMemcpyDeviceToHost, ss[i]), "D2H comp bound (sg)");

      // fence completion with an event
      cudaEvent_t ev; cuda_ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "mk event (c)");
      cuda_ck(cudaEventRecord(ev, ss[i]), "record event (c)");

      Job j; j.seq = next_seq++; j.stream = i; j.slot = s; j.raw_len = got; j.done = ev;
      inflight.push_back(j);
      next_slot[i] ^= 1;
    }

    try_flush();
  }

  // drain remaining jobs in order
  while (!inflight.empty()) {
    Job &j = inflight.front();
    cuda_ck(cudaEventSynchronize(j.done), "drain event (c)");
    cuda_ck(cudaEventDestroy(j.done), "destroy event (c)");
    size_t comp_len = *(host[j.stream].sz[j.slot].host);
    uint64_t r = j.raw_len;
    nvcz::fwrite_exact(&r, sizeof(r), stdout);
    nvcz::fwrite_exact(&comp_len, sizeof(comp_len), stdout);
    nvcz::fwrite_exact(comp_buffers[j.stream][j.slot].bytes(), comp_len, stdout);

    // Update progress counters
    output_bytes.fetch_add(sizeof(r) + sizeof(comp_len) + comp_len);
    chunks_processed.fetch_add(1);
    // device size buffers are pre-allocated per stream and reused
    write_seq++;
    inflight.pop_front();
  }

  // trailer
  uint64_t z=0; nvcz::fwrite_exact(&z,8,stdout); nvcz::fwrite_exact(&z,8,stdout);
  output_bytes.fetch_add(16); // Account for trailer

  for (int i=0;i<streams;++i){
    for (int s=0;s<SLOTS;++s){
      if (host[i].sz[s].host) cudaFreeHost(host[i].sz[s].host);
    }
    // free pre-allocated device size buffer (only allocated once per stream)
    if (host[i].sz[0].dev) {
      cudaFreeAsync(host[i].sz[0].dev, 0);
    }
  }

  // free persistent device buffers
  for (int i=0;i<streams;++i) {
    if (d_in[i])  cudaFree(d_in[i]);
    if (d_out[i]) cudaFree(d_out[i]);
  }

  // comp_buffers will be cleaned up automatically by destructor
  for (auto s : ss) cuda_ck(cudaStreamDestroy(s), "rm stream");

  // Stop progress reporting and show final statistics
  if (show_progress) {
    progress_running = false;
    progress_thread.join();

    // Show final statistics
    uint64_t final_input = input_bytes.load();
    uint64_t final_output = output_bytes.load();
    uint64_t final_chunks = chunks_processed.load();

    double in_mb = final_input / (1024.0 * 1024.0);
    double out_mb = final_output / (1024.0 * 1024.0);
    double ratio = final_input > 0 ? (double)final_output / final_input : 0.0;

    std::fprintf(stderr, "\nFinal: %.1f MB in, %.1f MB out (%.2fx), %lu chunks\n",
                in_mb, out_mb, ratio, final_chunks);
  }
}

// ---------------- pinned + overlapped single-GPU decompress ----------------

static void cmd_decompress(bool auto_tune, int cli_streams, size_t nvcomp_chunk_kb, bool show_progress, FILE* input_fp, bool enable_checksum)
{
  // Progress tracking
  std::atomic<uint64_t> input_bytes{0};
  std::atomic<uint64_t> output_bytes{0};
  std::atomic<uint64_t> chunks_processed{0};

  Header h{}; nvcz::fread_exact(&h, sizeof(h), stdin);
  input_bytes.fetch_add(sizeof(h));

  if (std::memcmp(h.magic, MAGIC, 5)!=0 || h.version!=1) die("bad header");
  auto algo = (Algo)h.algo;

  AutoTune t{}; if (auto_tune) t = pick_tuning(true);
  const uint32_t CHUNK_MB = h.chunk_mb; // honor file's chunk size
  const int streams = cli_streams > 0 ? cli_streams : (auto_tune ? t.streams : 3);

  auto codec = make_codec(algo, nvcomp_chunk_kb);
  if (!codec) die("codec not available");

  // Progress reporter thread
  std::thread progress_thread;
  std::atomic<bool> progress_running{true};

  if (show_progress) {
    progress_thread = std::thread([&input_fp, &progress_running, &input_bytes, &output_bytes, &chunks_processed, &enable_checksum]() {
      auto start_time = std::chrono::steady_clock::now();

      // Try to get input file size for percentage calculation
      uint64_t total_input_size = 0;
      bool has_total_size = false;
      // Note: input_fp is captured by reference in the lambda
      if (input_fp != stdin) {
        struct stat st;
        if (fstat(fileno(input_fp), &st) == 0) {
          total_input_size = st.st_size;
          has_total_size = true;
        }
      }

      while (progress_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Update every 0.5 seconds

        uint64_t in = input_bytes.load();
        uint64_t out = output_bytes.load();
        uint64_t chunks = chunks_processed.load();
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        if (elapsed > 0) {
          double ratio = out > 0 ? (double)in / out : 0.0;
          double speed_in = in / (1024.0 * 1024.0) / elapsed;
          double speed_out = out / (1024.0 * 1024.0) / elapsed;

          // Calculate percentage if we know the total size
          double percentage = 0.0;
          if (has_total_size && total_input_size > 0) {
            percentage = static_cast<double>(in) / total_input_size;
          }

          nvcz::render_progress_bar(percentage, in, has_total_size ? total_input_size : in, ratio, speed_in, speed_out, chunks, enable_checksum);
        }
      }
    });
  }

  const size_t CHUNK = size_t(CHUNK_MB) * 1024 * 1024;
  const int    SLOTS = 2;

  std::vector<cudaStream_t> ss(streams);
  for (int i=0;i<streams;++i) cuda_ck(cudaStreamCreate(&ss[i]), "mk d stream");

  struct Slots { PinnedBuffer raw[SLOTS]; PinnedBuffer comp[SLOTS]; };
  std::vector<Slots> host(streams);

  // initial caps; will grow per-block if needed
  for (int i=0;i<streams;++i) {
    for (int s=0;s<SLOTS;++s) {
      host[i].raw[s].alloc(CHUNK);
      host[i].comp[s].alloc(codec->max_compressed_bound(CHUNK)); // conservative
    }
  }

  // Persistent device buffers per stream for decode path
  const size_t WORST = codec->max_compressed_bound(CHUNK);
  std::vector<void*> d_in(streams, nullptr);
  std::vector<size_t> d_in_cap(streams, 0);
  std::vector<void*> d_out(streams, nullptr);
  for (int i=0;i<streams;++i) {
    // Start with worst-bound input capacity; will grow if needed
    cuda_ck(cudaMalloc(&d_in[i], WORST), "malloc d_in (sg d)");
    d_in_cap[i] = WORST;
    cuda_ck(cudaMalloc(&d_out[i], CHUNK), "malloc d_out (sg d)");
  }

  struct Job {
    uint64_t     seq = 0;
    int          stream = 0;
    int          slot = 0; // 0/1 ping-pong
    size_t       raw_len = 0;
    cudaEvent_t  done = nullptr;
  };

  std::vector<int> next_slot(streams, 0);
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

      nvcz::fwrite_exact(host[j.stream].raw[j.slot].bytes(), j.raw_len, stdout);

      // Update progress counters
      output_bytes.fetch_add(j.raw_len);
      chunks_processed.fetch_add(1);

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

      // Update input bytes for headers
      input_bytes.fetch_add(sizeof(r) + sizeof(c));

      int s = next_slot[i];
      if (host[i].raw[s].cap  < r) host[i].raw[s].alloc(r);
      if (host[i].comp[s].cap < c) host[i].comp[s].alloc(c);

      nvcz::fread_exact(host[i].comp[s].bytes(), c, stdin);

      // Update input bytes for compressed data
      input_bytes.fetch_add(c);

      // Ensure device input capacity
      if (d_in_cap[i] < c) {
        cudaFree(d_in[i]);
        cuda_ck(cudaMalloc(&d_in[i], c), "realloc d_in (sg d)");
        d_in_cap[i] = c;
      }

      // H2D compressed payload
      cuda_ck(cudaMemcpyAsync(d_in[i], host[i].comp[s].bytes(), c,
                              cudaMemcpyHostToDevice, ss[i]), "H2D comp (sg)");

      // Device-to-device decompress
      codec->decompress_dd(
        d_in[i], c,
        d_out[i],  r,
        ss[i]);

      // D2H raw into pinned buffer
      cuda_ck(cudaMemcpyAsync(host[i].raw[s].bytes(), d_out[i], r,
                              cudaMemcpyDeviceToHost, ss[i]), "D2H raw (sg)");

      cudaEvent_t ev; cuda_ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "mk event (d)");
      cuda_ck(cudaEventRecord(ev, ss[i]), "record event (d)");

      Job j; j.seq = next_seq++; j.stream = i; j.slot = s; j.raw_len = r; j.done = ev;
      inflight.push_back(j);

      next_slot[i] ^= 1;
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
    nvcz::fwrite_exact(host[j.stream].raw[j.slot].bytes(), j.raw_len, stdout);

    // Update progress counters for drained jobs
    output_bytes.fetch_add(j.raw_len);
    chunks_processed.fetch_add(1);

    write_seq++;
    inflight.pop_front();
  }

  for (auto s : ss) cuda_ck(cudaStreamDestroy(s), "rm d stream");

  // Free persistent device buffers
  for (int i=0;i<streams;++i) {
    if (d_in[i])  cudaFree(d_in[i]);
    if (d_out[i]) cudaFree(d_out[i]);
  }

  // Stop progress reporting and show final statistics
  if (show_progress) {
    progress_running = false;
    progress_thread.join();

    // Show final statistics
    uint64_t final_input = input_bytes.load();
    uint64_t final_output = output_bytes.load();
    uint64_t final_chunks = chunks_processed.load();

    double in_mb = final_input / (1024.0 * 1024.0);
    double out_mb = final_output / (1024.0 * 1024.0);
    double ratio = final_output > 0 ? (double)final_input / final_output : 0.0;

    std::fprintf(stderr, "\nFinal: %.1f MB in, %.1f MB out (%.2fx), %lu chunks\n",
                in_mb, out_mb, ratio, final_chunks);
  }
}

// ---------------- multi-GPU aware main (CLI with file support) ----------------

int main(int argc, char** argv)
{
  if (argc < 2) { usage(); return 1; }
  std::string mode = argv[1];
  std::string algo_str = "lz4";
  uint32_t chunk_mb = 32;
  bool auto_tune=false; int streams=0;
  size_t nvcomp_chunk_kb = 64;  // Default 64KB nvCOMP chunks

  // File I/O
  std::string input_file, output_file;
  std::vector<std::string> input_files;
  FILE* input_fp = stdin;
  FILE* output_fp = stdout;

  // Progress and statistics
  bool show_progress = false;

  // Checksum verification
  bool enable_checksum = false;

  // MGPU flags
  bool mgpu=false, auto_size=false;
  int streams_per_gpu_override=0;
  std::vector<int> gpu_ids_override;

  for (int i=2;i<argc;++i) {
    std::string a = argv[i];
    // Support --key=value forms
    auto starts_with = [&](const char* key){ return a.rfind(key, 0) == 0; };
    if (starts_with("--algo=")) { algo_str = a.substr(std::strlen("--algo=")); continue; }
    if (starts_with("--chunk-mb=")) { chunk_mb = std::max(1, std::atoi(a.substr(std::strlen("--chunk-mb=")).c_str())); continue; }
    if (starts_with("--nvcomp-chunk-kb=")) { nvcomp_chunk_kb = std::max(1UL, (size_t)std::atoi(a.substr(std::strlen("--nvcomp-chunk-kb=")).c_str())); continue; }
    if (starts_with("--streams=")) { streams = std::max(1, std::atoi(a.substr(std::strlen("--streams=")).c_str())); continue; }
    if (starts_with("--streams-per-gpu=")) { streams_per_gpu_override = std::max(1, std::atoi(a.substr(std::strlen("--streams-per-gpu=")).c_str())); continue; }
    if (starts_with("--gpus=")) {
      std::string list = a.substr(std::strlen("--gpus="));
      if (list != "all") {
        size_t pos=0;
        while (pos < list.size()) {
          size_t comma = list.find(',', pos);
          int id = std::atoi(list.substr(pos, (comma==std::string::npos? list.size():comma)-pos).c_str());
          gpu_ids_override.push_back(id);
          if (comma==std::string::npos) break; pos = comma+1;
        }
      }
      continue;
    }
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
    }
    else if (a=="-i" || a=="--input") {
      if (i+1<argc) {
        std::string file = argv[++i];
        if (input_file.empty()) {
          input_file = file;
        }
        input_files.push_back(file);
      } else { usage(); return 1; }
    }
    else if (starts_with("-i=")) {
      std::string file = a.substr(std::strlen("-i="));
      if (input_file.empty()) input_file = file; input_files.push_back(file);
    }
    else if (a=="-o" || a=="--output") {
      if (i+1<argc) {
        if (!output_file.empty()) { usage(); return 1; } // Only one output file allowed
        output_file = argv[++i];
      } else { usage(); return 1; }
    }
    else if (starts_with("-o=")) {
      if (!output_file.empty()) { usage(); return 1; }
      output_file = a.substr(std::strlen("-o="));
    }
    else if (a=="-p" || a=="--progress") { show_progress = true; }
    else if (a=="-c" || a=="--checksum") { enable_checksum = true; }
    else if (a=="--gds") { /* accept flag; handled later */ }
    else if (a[0] != '-' || a=="--") {
      // Non-option arguments are treated as input files
      input_files.push_back(a);
    }
    else { usage(); return 1; }
  }

  // Handle file I/O
  if (!input_files.empty()) {
    // Multiple input files - open the first one for now, we'll need to modify functions to handle multiple files
    input_fp = std::fopen(input_files[0].c_str(), "rb");
    if (!input_fp) {
      std::fprintf(stderr, "Error opening input file: %s\n", input_files[0].c_str());
      return 1;
    }
  }

  if (!output_file.empty()) {
    output_fp = std::fopen(output_file.c_str(), "wb");
    if (!output_fp) {
      std::fprintf(stderr, "Error opening output file: %s\n", output_file.c_str());
      if (input_fp != stdin) std::fclose(input_fp);
      return 1;
    }
  }

  // Redirect stdout/stdin if files are specified
  if (output_fp != stdout) {
    if (dup2(fileno(output_fp), fileno(stdout)) == -1) {
      std::fprintf(stderr, "Error redirecting stdout\n");
      return 1;
    }
  }
  if (input_fp != stdin) {
    if (dup2(fileno(input_fp), fileno(stdin)) == -1) {
      std::fprintf(stderr, "Error redirecting stdin\n");
      return 1;
    }
  }

  try {
    if (mode=="compress") {
      // If GDS requested via paths and flags
      bool want_gds = false;
      for (int i=2;i<argc;++i) if (std::string(argv[i]) == "--gds") want_gds = true;
      if (want_gds && !input_file.empty() && !output_file.empty() && !mgpu) {
        cmd_compress_gds(parse_algo(algo_str), chunk_mb, nvcomp_chunk_kb, show_progress, enable_checksum, input_file, output_file);
      } else if (mgpu) {
        AutoTune base = auto_tune ? pick_tuning(/*verbose=*/true)
                                  : AutoTune{chunk_mb, std::max(1, streams)};
        auto t = pick_mgpu_tuning(base, /*size_aware=*/auto_size,
                                  streams_per_gpu_override, gpu_ids_override);
        if (!auto_tune) t.chunk_mb = chunk_mb;
        if (want_gds && !input_file.empty() && !output_file.empty()) {
          // Reopen I/O as FILE* and pass to mgpu_gds (we already dup2'd stdin/stdout earlier)
          compress_mgpu_gds(parse_algo(algo_str), t, input_fp, output_fp, show_progress);
        } else {
          compress_mgpu(parse_algo(algo_str), t, input_fp, output_fp, show_progress);
        }
      } else {
        cmd_compress(parse_algo(algo_str), chunk_mb, auto_tune, streams, nvcomp_chunk_kb, show_progress, input_fp, enable_checksum);
      }
    } else if (mode=="decompress") {
      bool want_gds = false;
      for (int i=2;i<argc;++i) if (std::string(argv[i]) == "--gds") want_gds = true;
      if (want_gds && !input_file.empty() && !output_file.empty() && !mgpu) {
        cmd_decompress_gds(streams, nvcomp_chunk_kb, show_progress, input_file, output_file);
      } else if (mgpu) {
        AutoTune base = auto_tune ? pick_tuning(/*verbose=*/true)
                                  : AutoTune{chunk_mb, std::max(1, streams)};
        auto t = pick_mgpu_tuning(base, /*size_aware=*/auto_size,
                                  streams_per_gpu_override, gpu_ids_override);
        decompress_mgpu(t, input_fp, output_fp, show_progress);
      } else {
        cmd_decompress(auto_tune, streams, nvcomp_chunk_kb, show_progress, input_fp, enable_checksum);
      }
    } else {
      usage(); return 1;
    }
  } catch (...) {
    std::fprintf(stderr, "fatal: unhandled exception\n");
    return 2;
  }

  // Cleanup file handles
  if (input_fp != stdin) std::fclose(input_fp);
  if (output_fp != stdout) std::fclose(output_fp);

  return 0;
}
