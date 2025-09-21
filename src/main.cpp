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
#include <fstream>
#include <unistd.h>

using namespace nvcz;

static void usage() {
  std::fprintf(stderr,
    "nvcz: stream compressor using nvCOMP (LZ4, GDeflate, Snappy, Zstd)\n"
    "Usage:\n"
    "  nvcz compress --algo {lz4|gdeflate|snappy|zstd} [--chunk-mb N] [--nvcomp-chunk-kb N] [--auto] [--streams N]\n"
    "                [--mgpu] [--gpus all|0,2,3] [--streams-per-gpu N] [--auto-size]\n"
    "                [-i input_file] [-o output_file] [input_files...]\n"
    "  nvcz decompress [--auto] [--streams N]\n"
    "                  [--mgpu] [--gpus all|0,2,3] [--streams-per-gpu N] [--auto-size]\n"
    "                  [-i input_file] [-o output_file] [input_files...]\n"
    "Examples:\n"
    "  cat in.bin | nvcz compress --algo lz4 --auto > out.nvcz\n"
    "  nvcz decompress < out.nvcz > out.bin\n"
    "  nvcz compress --algo lz4 -i input.bin -o output.nvcz\n"
    "  nvcz decompress -i input.nvcz -o output.bin\n"
    "  nvcz compress --algo gdeflate --nvcomp-chunk-kb 256 --mgpu --gpus 0,1 -i big.bin -o big.nvcz\n");
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

      // use pre-allocated comp buffer (already sized to worst-case)

      // use pre-allocated device size buffer (reused per stream)

      // launch async compress on stream i; codec copies up to bound to comp, writes true size to dev size
      codec->compress_with_stream(
          host[i].raw[s].bytes(), got,
          comp_buffers[i][s].bytes(),
          ss[i],
          host[i].sz[0].dev);

      // also stage the size to host pinned
      cuda_ck(cudaMemcpyAsync(host[i].sz[s].host, host[i].sz[0].dev, sizeof(size_t),
                              cudaMemcpyDeviceToHost, ss[i]), "D2H comp size (sg)");

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
    // device size buffers are pre-allocated per stream and reused
    write_seq++;
    inflight.pop_front();
  }

  // trailer
  uint64_t z=0; nvcz::fwrite_exact(&z,8,stdout); nvcz::fwrite_exact(&z,8,stdout);

  for (int i=0;i<streams;++i){
    for (int s=0;s<SLOTS;++s){
      if (host[i].sz[s].host) cudaFreeHost(host[i].sz[s].host);
    }
    // free pre-allocated device size buffer (only allocated once per stream)
    if (host[i].sz[0].dev) {
      cudaFreeAsync(host[i].sz[0].dev, 0);
    }
  }

  // comp_buffers will be cleaned up automatically by destructor
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

      int s = next_slot[i];
      if (host[i].raw[s].cap  < r) host[i].raw[s].alloc(r);
      if (host[i].comp[s].cap < c) host[i].comp[s].alloc(c);

      nvcz::fread_exact(host[i].comp[s].bytes(), c, stdin);

      // launch async decomp
      codec->decompress_with_stream(
        host[i].comp[s].bytes(), c,
        host[i].raw[s].bytes(),  r,
        ss[i]);

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
    write_seq++;
    inflight.pop_front();
  }

  for (auto s : ss) cuda_ck(cudaStreamDestroy(s), "rm d stream");
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
    else if (a=="-o" || a=="--output") {
      if (i+1<argc) {
        if (!output_file.empty()) { usage(); return 1; } // Only one output file allowed
        output_file = argv[++i];
      } else { usage(); return 1; }
    }
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
      if (mgpu) {
        AutoTune base = auto_tune ? pick_tuning(/*verbose=*/true)
                                  : AutoTune{chunk_mb, std::max(1, streams)};
        auto t = pick_mgpu_tuning(base, /*size_aware=*/auto_size,
                                  streams_per_gpu_override, gpu_ids_override);
        if (!auto_tune) t.chunk_mb = chunk_mb;
        compress_mgpu(parse_algo(algo_str), t, input_fp, output_fp);
      } else {
        cmd_compress(parse_algo(algo_str), chunk_mb, auto_tune, streams, nvcomp_chunk_kb);
      }
    } else if (mode=="decompress") {
      if (mgpu) {
        AutoTune base = auto_tune ? pick_tuning(/*verbose=*/true)
                                  : AutoTune{chunk_mb, std::max(1, streams)};
        auto t = pick_mgpu_tuning(base, /*size_aware=*/auto_size,
                                  streams_per_gpu_override, gpu_ids_override);
        decompress_mgpu(t, input_fp, output_fp);
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

  // Cleanup file handles
  if (input_fp != stdin) std::fclose(input_fp);
  if (output_fp != stdout) std::fclose(output_fp);

  return 0;
}
