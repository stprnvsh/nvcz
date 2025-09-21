// src/mgpu.cpp
#include "nvcz/mgpu.hpp"   // Header, MAGIC, MgpuTune, Algo
#include "nvcz/util.hpp"   // die, cuda_ck, nv_ck, fread_exact/fwrite_exact, read_chunk_into_ptr
#include "nvcz/codec.hpp"
#include "nvcz/gds.hpp"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <unordered_map>
#include <atomic>
#include <vector>
#include <map>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>

namespace nvcz {

// ---------- pinned block & free-lists ----------

struct PinBlock {
  uint8_t* p = nullptr;
  size_t   cap = 0;
  size_t   n = 0;

  PinBlock() = default;
  PinBlock(const PinBlock&) = delete;
  PinBlock& operator=(const PinBlock&) = delete;
  PinBlock(PinBlock&& o) noexcept { move_from(o); }
  PinBlock& operator=(PinBlock&& o) noexcept { if (this!=&o){ release(); move_from(o);} return *this; }
  ~PinBlock(){ release(); }

  void move_from(PinBlock& o){ p=o.p; cap=o.cap; n=o.n; o.p=nullptr; o.cap=0; o.n=0; }

  void alloc_exact(size_t want){
    if (p && cap>=want) { n = 0; return; }
    release();
    void* tmp=nullptr;
    cuda_ck(cudaHostAlloc(&tmp, want, cudaHostAllocDefault), "cudaHostAlloc");
    p   = static_cast<uint8_t*>(tmp);
    cap = want;
    n   = 0;
  }

  void release(){
    if (p){ cudaFreeHost(p); p=nullptr; cap=0; n=0; }
  }
};

struct PinSize {
  size_t* p = nullptr;
  void alloc() {
    if (!p) {
      void* tmp=nullptr;
      cuda_ck(cudaHostAlloc(&tmp, sizeof(size_t), cudaHostAllocDefault), "cudaHostAlloc size");
      p = static_cast<size_t*>(tmp);
    }
  }
  PinSize() = default;
  PinSize(const PinSize&) = delete;
  PinSize& operator=(const PinSize&) = delete;
  PinSize(PinSize&& o) noexcept { p=o.p; o.p=nullptr; }
  PinSize& operator=(PinSize&& o) noexcept { if (this!=&o){ if(p) cudaFreeHost(p); p=o.p; o.p=nullptr; } return *this; }
  ~PinSize(){ if (p) cudaFreeHost(p); }
};

// simple blocking free-list
template <class T>
struct FreeList {
  std::mutex m;
  std::condition_variable cv;
  std::deque<T> q;

  void put(T&& v){ { std::lock_guard<std::mutex> lk(m); q.emplace_back(std::move(v)); } cv.notify_one(); }

  T get(){
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk,[&]{ return !q.empty(); });
    T v = std::move(q.front());
    q.pop_front();
    return v;
  }

  size_t size() const { std::lock_guard<std::mutex> lk(const_cast<std::mutex&>(m)); return q.size(); }
};

// ---------- job/result messages (carry events to keep async) ----------

struct JobC {                 // compress: host raw → host comp
  uint64_t idx = 0;
  PinBlock raw;               // from raw_free
};

struct ResC {
  uint64_t idx = 0;
  uint64_t raw_n = 0;
  PinBlock raw;               // return to raw_free after writer done
  PinBlock comp;              // owned; return to comp_free after writer done
  PinSize  comp_size_host;    // owned pinned host size_t buffer
  size_t*  d_comp_size = nullptr;    // reference to pre-allocated device-side size_t
  cudaEvent_t done = nullptr; // stream event marking completion
};

struct JobD {                 // decompress: host comp → host raw
  uint64_t idx = 0;
  uint64_t raw_n = 0;
  PinBlock comp;              // from comp_free (dispatcher fills)
};

struct ResD {
  uint64_t idx = 0;
  PinBlock comp;              // return to comp_free
  PinBlock raw;               // return to raw_free after writer done
  cudaEvent_t done = nullptr; // completion
};

// ---------- small MPMC queue ----------

template <class T>
struct Queue {
  std::mutex m;
  std::condition_variable cv;
  std::deque<T> q;
  bool closed=false;

  void push(T&& v){ { std::lock_guard<std::mutex> lk(m); if (closed) return; q.emplace_back(std::move(v)); } cv.notify_one(); }
  bool pop(T& out){
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk,[&]{return closed || !q.empty();});
    if (q.empty()) return false;
    out = std::move(q.front());
    q.pop_front();
    return true;
  }
  void close(){ { std::lock_guard<std::mutex> lk(m); closed=true; } cv.notify_all(); }
};

// ---------- helpers ----------

static bool stdin_is_regular_file() {
  struct stat st{}; if (fstat(STDIN_FILENO, &st) != 0) return false;
  return S_ISREG(st.st_mode);
}
static uint64_t stdin_file_size() {
  struct stat st{}; if (fstat(STDIN_FILENO, &st) != 0) return 0;
  return S_ISREG(st.st_mode) ? static_cast<uint64_t>(st.st_size) : 0;
}

std::vector<int> discover_gpus_ids() {
  int n=0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return {};
  std::vector<int> ids(n);
  for (int i=0;i<n;++i) ids[i]=i;
  return ids;
}

MgpuTune pick_mgpu_tuning(const AutoTune& base, bool size_aware,
                          int override_streams_per_gpu,
                          const std::vector<int>& gpu_ids_override)
{
  MgpuTune t{};
  t.chunk_mb   = base.chunk_mb;
  t.size_aware = size_aware;
  t.gpu_ids    = gpu_ids_override.empty() ? discover_gpus_ids() : gpu_ids_override;

  if (override_streams_per_gpu > 0) t.streams_per_gpu = override_streams_per_gpu;
  else                              t.streams_per_gpu = std::max(2, std::min(4, base.streams));

  if (t.size_aware && stdin_is_regular_file()) {
    auto sz = stdin_file_size();
    if (sz >= (2ull<<30)) t.streams_per_gpu = std::min(t.streams_per_gpu+1, 6);
    if (sz >= (8ull<<30)) t.streams_per_gpu = std::min(t.streams_per_gpu+1, 8);
  }
  return t;
}

// =====================================================================================
//                                         COMPRESS
// =====================================================================================

static void worker_compress(int gpu_id, int streams_per_gpu, Algo algo, size_t chunk_bytes,
                            Queue<JobC>& in, Queue<ResC>& out,
                            FreeList<PinBlock>& raw_free, FreeList<PinBlock>& comp_free,
                            std::atomic<bool>& any_error)
{
  try {
    cuda_ck(cudaSetDevice(gpu_id), "set device");
    std::vector<cudaStream_t> ss(streams_per_gpu);
    for (int i=0;i<streams_per_gpu;++i) cuda_ck(cudaStreamCreate(&ss[i]), "mk stream");

    auto codec = make_codec(algo, 64);  // Use default 64KB chunks for MGPU for now
    if (!codec) die("codec not available in worker");

    // Pre-allocate per-stream device size_t buffers (host size carried per-result)
    std::vector<size_t*> d_sizes(streams_per_gpu, nullptr);

    // Compute worst-case compressed size for this configured chunk size
    const size_t CHUNK_SIZE = chunk_bytes;
    const size_t WORST = codec->max_compressed_bound(CHUNK_SIZE);

    for (int i = 0; i < streams_per_gpu; ++i) {
      cuda_ck(cudaMallocAsync(reinterpret_cast<void**>(&d_sizes[i]), sizeof(size_t), ss[i]), "malloc d_comp_size per stream");
    }

    // Persistent device buffers per stream
    std::vector<void*> d_in(streams_per_gpu, nullptr);
    std::vector<void*> d_out(streams_per_gpu, nullptr);
    for (int i=0;i<streams_per_gpu;++i) {
      cuda_ck(cudaMalloc(&d_in[i], CHUNK_SIZE), "malloc d_in (mgpu c)");
      cuda_ck(cudaMalloc(&d_out[i], WORST), "malloc d_out (mgpu c)");
    }

    size_t lane = 0;
    JobC j;
    while (in.pop(j)) {
      // get output pinned block from global pool (no reuse until writer returns)
      int stream_idx = lane % ss.size();
      PinBlock comp = comp_free.get();
      if (comp.cap < WORST) comp.alloc_exact(WORST);

      cudaStream_t s = ss[stream_idx];

      // H2D into persistent device buffer
      cuda_ck(cudaMemcpyAsync(d_in[stream_idx], j.raw.p, j.raw.n, cudaMemcpyHostToDevice, s), "H2D raw (mgpu)");

      // compress async device-to-device; writes exact size to d_sizes[stream_idx]
      codec->compress_dd(d_in[stream_idx], j.raw.n, d_out[stream_idx], s, d_sizes[stream_idx]);

      // stage final size into a per-result pinned host size_t (owned by ResC)
      PinSize host_size;
      host_size.alloc();
      cuda_ck(cudaMemcpyAsync(host_size.p, d_sizes[stream_idx], sizeof(size_t), cudaMemcpyDeviceToHost, s), "D2H comp size");

      // D2H bound-sized payload into pinned comp buffer; writer will trim
      cuda_ck(cudaMemcpyAsync(comp.p, d_out[stream_idx], WORST, cudaMemcpyDeviceToHost, s), "D2H comp bound (mgpu)");

      // fence completion
      cudaEvent_t ev{};
      cuda_ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "evt create");
      cuda_ck(cudaEventRecord(ev, s), "evt record");

      // ship result - transfer ownership of pinned comp buffer to writer
      ResC r;
      r.idx           = j.idx;
      r.raw_n         = j.raw.n;
      r.raw           = std::move(j.raw);
      r.comp          = std::move(comp);
      r.comp_size_host= std::move(host_size);   // transfer ownership
      r.d_comp_size   = d_sizes[stream_idx];    // Reference to pre-allocated device size_t
      r.done          = ev;

      out.push(std::move(r));
      lane++;
    }

    // Cleanup per-stream buffers
    for (int i = 0; i < streams_per_gpu; ++i) {
      cudaFreeAsync(d_sizes[i], ss[i]);
      // Note: h_sizes will be cleaned up by destructor
    }

    for (int i=0;i<streams_per_gpu;++i) {
      if (d_in[i])  cudaFree(d_in[i]);
      if (d_out[i]) cudaFree(d_out[i]);
    }

    for (auto s: ss) cuda_ck(cudaStreamDestroy(s), "rm stream");
  } catch (...) {
    any_error.store(true);
    in.close();
  }
}

void compress_mgpu(Algo algo, const MgpuTune& t, FILE* input_fp, FILE* output_fp, bool show_progress)
{
  const size_t CHUNK = size_t(t.chunk_mb) * 1024 * 1024;
  const int    NGPU  = (int)t.gpu_ids.size();
  const int    RING_PER_GPU = std::max(3, t.streams_per_gpu * 2); // generous per-GPU ring

  // header via pwrite
  int out_fd = fileno(output_fp);
  Header h{}; std::memcpy(h.magic, MAGIC, 5);
  h.version = 1; h.algo = (uint8_t)algo; h.chunk_mb = t.chunk_mb;
  off_t out_off = 0;
  if (::pwrite(out_fd, &h, sizeof(h), out_off) != (ssize_t)sizeof(h)) die("pwrite header (mgpu gds)");
  out_off += sizeof(h);

  // compute worst bound once
  auto bound_codec = make_codec(algo, 64);  // Use default 64KB chunks for MGPU for now
  if (!bound_codec) die("codec not available (compress bound)");
  const size_t WORST = bound_codec->max_compressed_bound(CHUNK);

  // Progress tracking
  std::atomic<uint64_t> input_bytes{0};
  std::atomic<uint64_t> output_bytes{0};
  std::atomic<uint64_t> chunks_processed{0};

  // global free-lists
  FreeList<PinBlock> raw_free;
  FreeList<PinBlock> comp_free;

  // pre-alloc pinned buffers
  const int RAW_POOL  = std::max(1, NGPU) * RING_PER_GPU;
  const int COMP_POOL = std::max(1, NGPU) * RING_PER_GPU;

  for (int i=0;i<RAW_POOL;  ++i){ PinBlock b; b.alloc_exact(CHUNK); raw_free.put(std::move(b)); }
  for (int i=0;i<COMP_POOL; ++i){ PinBlock b; b.alloc_exact(WORST);  comp_free.put(std::move(b)); }

  Queue<JobC> q_jobs;
  Queue<ResC> q_results;
  std::atomic<bool> any_error{false};

  // Progress reporter thread
  std::thread progress_thread;
  std::atomic<bool> progress_running{true};

  if (show_progress) {
    progress_thread = std::thread([&]() {
      auto start_time = std::chrono::steady_clock::now();

      // Try to get input file size for percentage calculation
      uint64_t total_input_size = 0;
      bool has_total_size = false;
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

          nvcz::render_progress_bar(percentage, in, has_total_size ? total_input_size : in, ratio, speed_in, speed_out, chunks, true); // Test checksum display
        }
      }
    });
  }

  // workers
  std::vector<std::thread> workers;
  for (int gpu_id : t.gpu_ids) {
    workers.emplace_back(worker_compress, gpu_id, t.streams_per_gpu, algo, CHUNK,
                         std::ref(q_jobs), std::ref(q_results),
                         std::ref(raw_free), std::ref(comp_free),
                         std::ref(any_error));
  }

  // dispatcher: read stdin into raw pinned blocks
  std::thread dispatcher([&](){
    uint64_t idx=0;
    for (;;) {
      PinBlock raw = raw_free.get();
      size_t got = read_chunk_into_ptr(raw.p, CHUNK, input_fp);
      raw.n = got;
      if (got == 0) { // return empty and stop
        raw_free.put(std::move(raw));
        break;
      }
      input_bytes.fetch_add(got);
      JobC j; j.idx = idx++; j.raw = std::move(raw);
      q_jobs.push(std::move(j));
    }
    q_jobs.close();
  });

  // writer: maintain order, wait only for next event, then read exact size and emit
  std::thread writer([&](){
    uint64_t next = 0;
    std::map<uint64_t, ResC> hold;
    ResC r;

    auto try_flush = [&](){
      while (true) {
        auto it = hold.find(next);
        if (it == hold.end()) break;
        ResC& rr = it->second;

        // wait for GPU job to finish
        cuda_ck(cudaEventSynchronize(rr.done), "evt sync");
        cudaEventDestroy(rr.done);

        // read the exact size
        const size_t comp_len = *(rr.comp_size_host.p);
        const uint64_t raw_len = rr.raw_n;

        // frame + payload
        fwrite_exact(&raw_len,  sizeof(raw_len),  output_fp);
        fwrite_exact(&comp_len, sizeof(comp_len), output_fp);
        fwrite_exact(rr.comp.p, comp_len, output_fp);

        // Update progress counters
        output_bytes.fetch_add(sizeof(raw_len) + sizeof(comp_len) + comp_len);
        chunks_processed.fetch_add(1);

        // recycle buffers to pools
        raw_free.put(std::move(rr.raw));
        comp_free.put(std::move(rr.comp));

        hold.erase(it);
        next++;
      }
    };

    while (q_results.pop(r)) {
      hold.emplace(r.idx, std::move(r));
      try_flush();
    }

    // trailer
    uint64_t z=0; fwrite_exact(&z,8,output_fp); fwrite_exact(&z,8,output_fp);
    output_bytes.fetch_add(16); // Account for trailer
    try_flush();
  });

  dispatcher.join();
  for (auto& th : workers) th.join();
  q_results.close();
  writer.join();

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

  if (any_error.load()) die("MGPU compress failed");
}

void compress_mgpu_gds(Algo algo, const MgpuTune& t, FILE* input_fp, FILE* output_fp, bool show_progress)
{
  const size_t CHUNK = size_t(t.chunk_mb) * 1024 * 1024;
  const int    NGPU  = (int)t.gpu_ids.size();
  const int    RING_PER_GPU = std::max(3, t.streams_per_gpu * 2);

  // header
  Header h{}; std::memcpy(h.magic, MAGIC, 5);
  h.version = 1; h.algo = (uint8_t)algo; h.chunk_mb = t.chunk_mb;
  fwrite_exact(&h, sizeof(h), output_fp);

  // compute worst bound once
  auto bound_codec = make_codec(algo, 64);
  if (!bound_codec) die("codec not available (compress bound)");
  const size_t WORST = bound_codec->max_compressed_bound(CHUNK);

  // Open GDS on stdin/stdout descriptors if they are regular files
  int in_fd = fileno(input_fp);
  int out_fd = fileno(output_fp);
  nvcz::GDSFile gds_in;  bool gds_in_ok  = gds_in.open_fd(in_fd);
  nvcz::GDSFile gds_out; bool gds_out_ok = gds_out.open_fd(out_fd);
  if (!gds_in_ok || !gds_out_ok) die("GDS open failed (mgpu)");

  // Progress tracking
  std::atomic<uint64_t> input_bytes{0};
  std::atomic<uint64_t> output_bytes{0};
  std::atomic<uint64_t> chunks_processed{0};

  // workers
  struct JobC { uint64_t idx=0; size_t n=0; int lane=0; off_t file_off=0; };
  struct ResC { uint64_t idx=0; uint64_t raw_n=0; PinSize comp_size_host; void* d_comp_dev=nullptr; size_t* d_comp_size=nullptr; cudaEvent_t done=nullptr; int lane=0; };

  Queue<JobC> q_jobs;
  Queue<ResC> q_results;
  std::atomic<bool> any_error{false};

  // No host comp pool needed for GDS path (zero-copy device->file)

  // Launch per-GPU workers
  std::vector<std::thread> workers;
  for (int gpu_id : t.gpu_ids) {
    workers.emplace_back([&, gpu_id]{
      try {
        cuda_ck(cudaSetDevice(gpu_id), "set device");
        auto codec = make_codec(algo, 64);
        if (!codec) die("codec not available in worker");

        std::vector<cudaStream_t> ss(t.streams_per_gpu);
        for (int i=0;i<t.streams_per_gpu;++i) cuda_ck(cudaStreamCreate(&ss[i]), "mk stream");

        const size_t CHUNK_SIZE = CHUNK;
        const size_t BOUND = codec->max_compressed_bound(CHUNK_SIZE);

        // per-stream device buffers
        std::vector<void*> d_in(t.streams_per_gpu, nullptr);
        // d_out allocated per job (freed by writer)
        std::vector<size_t*> d_sizes(t.streams_per_gpu, nullptr);
        for (int i=0;i<t.streams_per_gpu;++i){
          cuda_ck(cudaMalloc(&d_in[i],  CHUNK_SIZE), "mgpu gds d_in");
          cuda_ck(cudaMallocAsync((void**)&d_sizes[i], sizeof(size_t), ss[i]), "mgpu gds d_sz");
        }

        JobC j;
        while (q_jobs.pop(j)) {
          int lane = j.lane % ss.size();
          cudaStream_t s = ss[lane];

          // Read from file directly into device buffer
          ssize_t got = gds_in.read_to_gpu(d_in[lane], j.n, j.file_off);
          if (got != (ssize_t)j.n) die("GDS read (mgpu)");
          input_bytes.fetch_add(j.n);

          // Allocate per-job device output buffer
          void* d_out_job = nullptr;
          cuda_ck(cudaMalloc(&d_out_job, BOUND), "mgpu gds d_out job");

          // Compress device-to-device
          codec->compress_dd(d_in[lane], j.n, d_out_job, s, d_sizes[lane]);

          // stage size to host
          PinSize hs; hs.alloc();
          cuda_ck(cudaMemcpyAsync(hs.p, d_sizes[lane], sizeof(size_t), cudaMemcpyDeviceToHost, s), "D2H size (mgpu)");

          // event
          cudaEvent_t ev; cuda_ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "evt");
          cuda_ck(cudaEventRecord(ev, s), "rec");

          ResC r; r.idx=j.idx; r.raw_n=j.n; r.comp_size_host=std::move(hs); r.d_comp_dev=d_out_job; r.d_comp_size=d_sizes[lane]; r.done=ev; r.lane=lane;
          q_results.push(std::move(r));
        }

        for (int i=0;i<t.streams_per_gpu;++i){
          if (d_in[i]) cudaFree(d_in[i]);
          if (d_sizes[i]) cudaFreeAsync(d_sizes[i], ss[i]);
          cudaStreamDestroy(ss[i]);
        }
      } catch (...) {
        any_error.store(true);
        q_jobs.close();
      }
    });
  }

  // Writer thread: emit frames to output_fp using cuFile from device buffer
  std::thread writer([&]{
    uint64_t next=0;
    std::map<uint64_t, ResC> hold;
    ResC r;
    off_t writer_off = sizeof(Header); // continue after header
    while (q_results.pop(r)) { hold.emplace(r.idx, std::move(r));
      while (true) {
        auto it = hold.find(next); if (it==hold.end()) break; ResC &rr = it->second;
        cuda_ck(cudaEventSynchronize(rr.done), "sync"); cudaEventDestroy(rr.done);
        size_t comp_len = *(rr.comp_size_host.p);
        uint64_t raw_len = rr.raw_n;
        // Write frame header and payload using pwrite (headers) + fwrite payload
        if (::pwrite(out_fd, &raw_len, sizeof(raw_len), writer_off) != (ssize_t)sizeof(raw_len)) die("pwrite r (mgpu gds)");
        writer_off += sizeof(raw_len);
        if (::pwrite(out_fd, &comp_len, sizeof(comp_len), writer_off) != (ssize_t)sizeof(comp_len)) die("pwrite c (mgpu gds)");
        writer_off += sizeof(comp_len);
        // payload: direct GPU->file write via GDS (zero-copy)
        ssize_t w = gds_out.write_from_gpu(rr.d_comp_dev, comp_len, writer_off);
        if (w != (ssize_t)comp_len) die("cuFileWrite (mgpu gds)");
        // free per-job device buffer after event
        cuda_ck(cudaFree(rr.d_comp_dev), "free d_out job");
        writer_off += comp_len;
        output_bytes.fetch_add(sizeof(raw_len)+sizeof(comp_len)+comp_len);
        chunks_processed.fetch_add(1);
        hold.erase(it); next++;
      }
    }
    uint64_t z=0; fwrite_exact(&z,8,output_fp); fwrite_exact(&z,8,output_fp);
    output_bytes.fetch_add(16);
  });

  // Dispatcher: enumerate file offsets and send jobs of size CHUNK
  std::thread dispatcher([&]{
    off_t off=0; uint64_t idx=0;
    while (true) {
      // Probe read amount by reading from CPU side just to detect EOF quickly
      // For simplicity, use fstat to get size and compute number of chunks
      struct stat st{}; if (fstat(fileno(input_fp), &st)!=0) die("stat (mgpu gds)");
      uint64_t total = st.st_size;
      if (off >= (off_t)total) break;
      size_t n = std::min<size_t>(CHUNK, total - off);
      JobC j; j.idx=idx++; j.n=n; j.file_off=off; j.lane=(int)(idx % (NGPU*t.streams_per_gpu));
      q_jobs.push(std::move(j));
      off += n;
    }
    q_jobs.close();
  });

  dispatcher.join();
  for (auto& th: workers) th.join();
  q_results.close();
  writer.join();

  if (any_error.load()) die("MGPU GDS compress failed");
}

// =====================================================================================
//                                        DECOMPRESS
// =====================================================================================

static void worker_decompress(int gpu_id, int streams_per_gpu, Algo algo,
                              Queue<JobD>& in, Queue<ResD>& out,
                              FreeList<PinBlock>& raw_free, FreeList<PinBlock>& comp_free,
                              std::atomic<bool>& any_error)
{
  try {
    cuda_ck(cudaSetDevice(gpu_id), "set device");
    std::vector<cudaStream_t> ss(streams_per_gpu);
    for (int i=0;i<streams_per_gpu;++i) cuda_ck(cudaStreamCreate(&ss[i]), "mk d stream");

    auto codec = make_codec(algo, 64);  // Use default 64KB chunks for MGPU for now
    if (!codec) die("codec not available in d worker");

    // Persistent device buffers sized on demand per stream
    std::vector<void*> d_in(streams_per_gpu, nullptr);
    std::vector<size_t> d_in_cap(streams_per_gpu, 0);
    std::vector<void*> d_out(streams_per_gpu, nullptr);
    std::vector<size_t> d_out_cap(streams_per_gpu, 0);

    size_t lane=0;
    JobD j;
    while (in.pop(j)) {
      // get a raw output pinned block from pool (exact size)
      PinBlock raw = raw_free.get();
      if (raw.cap < j.raw_n) { raw.alloc_exact(j.raw_n); }
      raw.n = j.raw_n;

      cudaStream_t s = ss[lane % ss.size()];

      // Ensure persistent device buffer capacities
      int stream_idx = lane % ss.size();
      if (d_in_cap[stream_idx] < j.comp.n) {
        if (d_in[stream_idx]) cudaFree(d_in[stream_idx]);
        cuda_ck(cudaMalloc(&d_in[stream_idx], j.comp.n), "malloc d_in (mgpu d)");
        d_in_cap[stream_idx] = j.comp.n;
      }
      if (d_out_cap[stream_idx] < j.raw_n) {
        if (d_out[stream_idx]) cudaFree(d_out[stream_idx]);
        cuda_ck(cudaMalloc(&d_out[stream_idx], j.raw_n), "malloc d_out (mgpu d)");
        d_out_cap[stream_idx] = j.raw_n;
      }

      // H2D comp data into persistent device buffer
      cuda_ck(cudaMemcpyAsync(d_in[stream_idx], j.comp.p, j.comp.n, cudaMemcpyHostToDevice, s), "H2D comp (mgpu)");

      // Device-to-device decompress
      codec->decompress_dd(d_in[stream_idx], j.comp.n, d_out[stream_idx], j.raw_n, s);

      // D2H raw payload into pinned host
      cuda_ck(cudaMemcpyAsync(raw.p, d_out[stream_idx], j.raw_n, cudaMemcpyDeviceToHost, s), "D2H raw (mgpu)");

      cudaEvent_t ev{};
      cuda_ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "evt create");
      cuda_ck(cudaEventRecord(ev, s), "evt record");

      ResD r;
      r.idx  = j.idx;
      r.comp = std::move(j.comp); // returned after write
      r.raw  = std::move(raw);    // returned after write
      r.done = ev;

      out.push(std::move(r));
      lane++;
    }

    for (auto s: ss) cuda_ck(cudaStreamDestroy(s), "rm d stream");

    for (int i=0;i<streams_per_gpu;++i) {
      if (d_in[i])  cudaFree(d_in[i]);
      if (d_out[i]) cudaFree(d_out[i]);
    }
  } catch (...) {
    any_error.store(true);
    in.close();
  }
}

void decompress_mgpu(const MgpuTune& t, FILE* input_fp, FILE* output_fp, bool show_progress)
{
  // Progress tracking
  std::atomic<uint64_t> input_bytes{0};
  std::atomic<uint64_t> output_bytes{0};
  std::atomic<uint64_t> chunks_processed{0};

  // header
  Header h{}; fread_exact(&h, sizeof(h), input_fp);
  input_bytes.fetch_add(sizeof(h));
  if (std::memcmp(h.magic, MAGIC, 5)!=0 || h.version!=1) die("bad header");
  Algo algo = (Algo)h.algo;

  const size_t CHUNK = size_t(h.chunk_mb) * 1024 * 1024;
  const int    NGPU  = (int)t.gpu_ids.size();
  const int    RING_PER_GPU = std::max(3, t.streams_per_gpu * 2);

  // conservative worst bound for comp pool (enough to fetch comp blocks)
  auto bound_codec = make_codec(algo, 64);  // Use default 64KB chunks for MGPU for now
  if (!bound_codec) die("codec not available (decompress bound)");
  const size_t WORST = bound_codec->max_compressed_bound(CHUNK);

  // free-lists
  FreeList<PinBlock> raw_free;
  FreeList<PinBlock> comp_free;

  // pre-alloc pools
  const int RAW_POOL  = std::max(1, NGPU) * RING_PER_GPU;
  const int COMP_POOL = std::max(1, NGPU) * RING_PER_GPU;
  for (int i=0;i<RAW_POOL;  ++i){ PinBlock b; b.alloc_exact(CHUNK); raw_free.put(std::move(b)); }
  for (int i=0;i<COMP_POOL; ++i){ PinBlock b; b.alloc_exact(WORST);  comp_free.put(std::move(b)); }

  Queue<JobD> q_jobs;
  Queue<ResD> q_results;
  std::atomic<bool> any_error{false};

  // workers
  std::vector<std::thread> workers;
  for (int gpu_id : t.gpu_ids) {
    workers.emplace_back(worker_decompress, gpu_id, t.streams_per_gpu, algo,
                         std::ref(q_jobs), std::ref(q_results),
                         std::ref(raw_free), std::ref(comp_free),
                         std::ref(any_error));
  }

  // Progress reporter thread
  std::thread progress_thread;
  std::atomic<bool> progress_running{true};

  if (show_progress) {
    progress_thread = std::thread([&]() {
      auto start_time = std::chrono::steady_clock::now();

      // Try to get input file size for percentage calculation
      uint64_t total_input_size = 0;
      bool has_total_size = false;
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

          nvcz::render_progress_bar(percentage, in, has_total_size ? total_input_size : in, ratio, speed_in, speed_out, chunks, true); // Test checksum display
        }
      }
    });
  }

  // dispatcher: read (r,c,data)
  std::thread dispatcher([&](){
    uint64_t idx=0;
    for (;;) {
      uint64_t r=0, c=0;
      size_t got_r = std::fread(&r,1,sizeof(r),input_fp);
      size_t got_c = std::fread(&c,1,sizeof(c),input_fp);
      if (got_r != sizeof(r) || got_c != sizeof(c)) die("truncated header");
      if (r==0 && c==0) break;

      input_bytes.fetch_add(sizeof(r) + sizeof(c));

      PinBlock comp = comp_free.get();
      if (comp.cap < c) comp.alloc_exact(c);
      fread_exact(comp.p, c, input_fp);
      comp.n = c;
      input_bytes.fetch_add(c);

      JobD j; j.idx = idx++; j.raw_n = r; j.comp = std::move(comp);
      q_jobs.push(std::move(j));
    }
    q_jobs.close();
  });

  // writer: in-order, event-driven
  std::thread writer([&](){
    uint64_t next=0;
    std::map<uint64_t, ResD> hold;
    ResD r;

    auto try_flush = [&](){
      while (true) {
        auto it = hold.find(next);
        if (it == hold.end()) break;
        ResD& rr = it->second;
        cuda_ck(cudaEventSynchronize(rr.done), "evt sync");
        cudaEventDestroy(rr.done);

        fwrite_exact(rr.raw.p, rr.raw.n, output_fp);

        // Update progress counters
        output_bytes.fetch_add(rr.raw.n);
        chunks_processed.fetch_add(1);

        comp_free.put(std::move(rr.comp));
        raw_free.put(std::move(rr.raw));

        hold.erase(it);
        next++;
      }
    };

    while (q_results.pop(r)) {
      hold.emplace(r.idx, std::move(r));
      try_flush();
    }
    try_flush();
  });

  dispatcher.join();
  for (auto& th : workers) th.join();
  q_results.close();
  writer.join();

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

  if (any_error.load()) die("MGPU decompress failed");
}

} // namespace nvcz
