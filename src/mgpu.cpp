// src/mgpu.cpp
#include "nvcz/mgpu.hpp"   // Header, MAGIC, MgpuTune, Algo
#include "nvcz/util.hpp"   // die, cuda_ck, nv_ck, fread_exact/fwrite_exact, read_chunk_into_ptr
#include "nvcz/codec.hpp"

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
#include <functional>
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
  PinBlock* comp = nullptr;   // pointer to pre-allocated buffer
  size_t*  comp_size_host = nullptr; // reference to pre-allocated pinned host size_t
  size_t*  d_comp_size = nullptr; // reference to pre-allocated device-side size_t
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

static void worker_compress(int gpu_id, int streams_per_gpu, Algo algo,
                            Queue<JobC>& in, Queue<ResC>& out,
                            FreeList<PinBlock>& raw_free,
                            std::atomic<bool>& any_error)
{
  try {
    cuda_ck(cudaSetDevice(gpu_id), "set device");
    std::vector<cudaStream_t> ss(streams_per_gpu);
    for (int i=0;i<streams_per_gpu;++i) cuda_ck(cudaStreamCreate(&ss[i]), "mk stream");

    auto codec = make_codec(algo, 64);  // Use default 64KB chunks for MGPU for now
    if (!codec) die("codec not available in worker");

    // Pre-allocate per-stream pinned size_t and device size_t buffers
    std::vector<PinSize> h_sizes(streams_per_gpu);
    std::vector<size_t*> d_sizes(streams_per_gpu, nullptr);
    std::vector<PinBlock> comp_buffers(streams_per_gpu * 2);  // 2 per stream for ping-pong

    // Compute worst-case compressed size for this chunk size
    const size_t CHUNK_SIZE = 32ULL * 1024 * 1024;  // Default chunk size used by autotune
    const size_t WORST = codec->max_compressed_bound(CHUNK_SIZE);

    for (int i = 0; i < streams_per_gpu; ++i) {
      h_sizes[i].alloc();  // Pre-allocate pinned host size_t
      cuda_ck(cudaMallocAsync(reinterpret_cast<void**>(&d_sizes[i]), sizeof(size_t), ss[i]), "malloc d_comp_size per stream");
      // Pre-allocate compressed buffers per stream (no more dynamic allocation)
      for (int j = 0; j < 2; ++j) {
        comp_buffers[i * 2 + j].alloc_exact(WORST);
      }
    }

    size_t lane = 0;
    JobC j;
    while (in.pop(j)) {
      // get output pinned block (use pre-allocated buffer, no allocation)
      int stream_idx = lane % ss.size();
      int buffer_idx = (lane / ss.size()) % 2;  // Alternate between 2 buffers per stream
      PinBlock& comp = comp_buffers[stream_idx * 2 + buffer_idx];

      cudaStream_t s = ss[stream_idx];

      // compress async; codec copies D->H bound bytes to comp.p and writes exact size to d_comp_size
      codec->compress_with_stream(j.raw.p, j.raw.n, comp.p, s, d_sizes[stream_idx]);

      // stage final size into pinned host size_t
      cuda_ck(cudaMemcpyAsync(h_sizes[stream_idx].p, d_sizes[stream_idx], sizeof(size_t), cudaMemcpyDeviceToHost, s), "D2H comp size");

      // fence completion
      cudaEvent_t ev{};
      cuda_ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "evt create");
      cuda_ck(cudaEventRecord(ev, s), "evt record");

      // ship result - reference pre-allocated buffers, don't own them
      ResC r;
      r.idx           = j.idx;
      r.raw_n         = j.raw.n;
      r.raw           = std::move(j.raw);
      r.comp          = &comp;  // Pointer to pre-allocated buffer
      r.comp_size_host= h_sizes[stream_idx].p;  // Reference to pre-allocated pinned host size_t
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

    for (auto s: ss) cuda_ck(cudaStreamDestroy(s), "rm stream");
  } catch (...) {
    any_error.store(true);
    in.close();
  }
}

void compress_mgpu(Algo algo, const MgpuTune& t)
{
  compress_mgpu_with_files(algo, t, &std::cin, &std::cout);
}

void compress_mgpu_with_files(Algo algo, const MgpuTune& t, std::istream* input_file, std::ostream* output_file,
                             std::function<void(size_t, size_t)> progress_callback, size_t total_size)
{
  const size_t CHUNK = size_t(t.chunk_mb) * 1024 * 1024;
  const int    NGPU  = (int)t.gpu_ids.size();
  const int    RING_PER_GPU = std::max(3, t.streams_per_gpu * 2); // generous per-GPU ring

  // header
  Header h{}; std::memcpy(h.magic, MAGIC, 5);
  h.version = 1; h.algo = (uint8_t)algo; h.chunk_mb = t.chunk_mb;
  fwrite_exact(&h, sizeof(h), output_file);

  // compute worst bound once
  auto bound_codec = make_codec(algo, 64);  // Use default 64KB chunks for MGPU for now
  if (!bound_codec) die("codec not available (compress bound)");
  const size_t WORST = bound_codec->max_compressed_bound(CHUNK);

  // global free-lists (comp_free removed since we pre-allocate per-stream)
  FreeList<PinBlock> raw_free;

  // pre-alloc pinned buffers (comp pools removed - pre-allocated per-stream)
  const int RAW_POOL  = std::max(1, NGPU) * RING_PER_GPU;

  for (int i=0;i<RAW_POOL;  ++i){ PinBlock b; b.alloc_exact(CHUNK); raw_free.put(std::move(b)); }

  Queue<JobC> q_jobs;
  Queue<ResC> q_results;
  std::atomic<bool> any_error{false};

  // workers
  std::vector<std::thread> workers;
  for (int gpu_id : t.gpu_ids) {
    workers.emplace_back(worker_compress, gpu_id, t.streams_per_gpu, algo,
                         std::ref(q_jobs), std::ref(q_results),
                         std::ref(raw_free),
                         std::ref(any_error));
  }

  // dispatcher: read from input file into raw pinned blocks
  std::thread dispatcher([&](){
    uint64_t idx=0;
    for (;;) {
      PinBlock raw = raw_free.get();

      // Use read_chunk_into_ptr with custom input file if provided
      size_t got = read_chunk_into_ptr(raw.p, CHUNK, input_file);
      raw.n = got;
      if (got == 0) { // return empty and stop
        raw_free.put(std::move(raw));
        break;
      }
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
        const size_t comp_len = *(rr.comp_size_host);
        const uint64_t raw_len = rr.raw_n;

        // frame + payload
        fwrite_exact(&raw_len,  sizeof(raw_len),  output_file);
        fwrite_exact(&comp_len, sizeof(comp_len), output_file);
        fwrite_exact(rr.comp->p, comp_len, output_file);

        // recycle buffers - comp buffer is pre-allocated per-stream and reused
        // Note: rr.d_comp_size and rr.comp_size_host are now pre-allocated per-stream and reused
        // Note: rr.comp is also pre-allocated per-stream and reused (no need to return to pool)
        raw_free.put(std::move(rr.raw));

        hold.erase(it);
        next++;

        // Progress callback
        if (progress_callback) {
          static size_t total_processed = 0;
          total_processed += raw_len;
          progress_callback(total_processed, total_size);
        }
      }
    };

    while (q_results.pop(r)) {
      hold.emplace(r.idx, std::move(r));
      try_flush();
    }

    // trailer
    uint64_t z=0; fwrite_exact(&z,8,output_file); fwrite_exact(&z,8,output_file);
    try_flush();
  });

  dispatcher.join();
  for (auto& th : workers) th.join();
  q_results.close();
  writer.join();

  if (any_error.load()) die("MGPU compress failed");
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

    size_t lane=0;
    JobD j;
    while (in.pop(j)) {
      // get a raw output pinned block from pool (exact size)
      PinBlock raw = raw_free.get();
      if (raw.cap < j.raw_n) { raw.alloc_exact(j.raw_n); }
      raw.n = j.raw_n;

      cudaStream_t s = ss[lane % ss.size()];

      codec->decompress_with_stream(j.comp.p, j.comp.n, raw.p, j.raw_n, s);

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
  } catch (...) {
    any_error.store(true);
    in.close();
  }
}

void decompress_mgpu(const MgpuTune& t)
{
  decompress_mgpu_with_files(t, &std::cin, &std::cout);
}

void decompress_mgpu_with_files(const MgpuTune& t, std::istream* input_file, std::ostream* output_file,
                               std::function<void(size_t, size_t)> progress_callback, size_t total_size)
{
  // header
  Header h{}; fread_exact(&h, sizeof(h), input_file);
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

  // dispatcher: read (r,c,data)
  std::thread dispatcher([&](){
    uint64_t idx=0;
    for (;;) {
      uint64_t r=0, c=0;
      input_file->read(reinterpret_cast<char*>(&r), sizeof(r));
      input_file->read(reinterpret_cast<char*>(&c), sizeof(c));
      size_t got_r = input_file->gcount();
      size_t got_c = input_file->gcount();
      if (got_r != sizeof(r) || got_c != sizeof(c)) die("truncated header");
      if (r==0 && c==0) break;

      PinBlock comp = comp_free.get();
      if (comp.cap < c) comp.alloc_exact(c);
      fread_exact(comp.p, c, input_file);
      comp.n = c;

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

        fwrite_exact(rr.raw.p, rr.raw.n, output_file);

        comp_free.put(std::move(rr.comp));
        raw_free.put(std::move(rr.raw));

        hold.erase(it);
        next++;

        // Progress callback
        if (progress_callback) {
          static size_t total_processed = 0;
          total_processed += rr.raw.n;
          progress_callback(total_processed, total_size);
        }
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

  if (any_error.load()) die("MGPU decompress failed");
}

} // namespace nvcz
