#pragma once
#include <cuda_runtime.h>
#include <nvcomp/shared_types.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <string>
#include <sys/stat.h>

namespace nvcz {

// ----- error + abort -----
inline void die(const char* msg) {
  std::fprintf(stderr, "%s\n", msg);
  std::quick_exit(2);
}

inline void cuda_ck(cudaError_t st, const char* what) {
  if (st != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s: %s\n", what, cudaGetErrorString(st));
    std::quick_exit(3);
  }
}
inline void nv_ck(nvcompStatus_t st, const char* what) {
  if (st != nvcompSuccess) {
    std::fprintf(stderr, "nvCOMP error: %s: %d\n", what, (int)st);
    std::quick_exit(3);
  }
}

// ----- small device helper -----
template <typename T>
T* d_alloc_and_copy(const T* h, size_t count) {
  T* d=nullptr;
  cuda_ck(cudaMalloc(&d, sizeof(T)*count), "cudaMalloc d_alloc_and_copy");
  cuda_ck(cudaMemcpy(d, h, sizeof(T)*count, cudaMemcpyHostToDevice),
          "cudaMemcpy H2D d_alloc_and_copy");
  return d;
}

// ----- pinned host buffer -----
struct PinnedBuffer {
  void*  ptr = nullptr;
  size_t cap = 0;

  void alloc(size_t n) {
    if (ptr && cap >= n) return;
    free();
    cuda_ck(cudaHostAlloc(&ptr, n, cudaHostAllocDefault), "cudaHostAlloc");
    cap = n;
  }
  uint8_t* bytes() { return static_cast<uint8_t*>(ptr); }
  const uint8_t* bytes() const { return static_cast<const uint8_t*>(ptr); }
  ~PinnedBuffer() { free(); }
  void free() {
    if (ptr) { cudaFreeHost(ptr); ptr = nullptr; cap = 0; }
  }
};

// ----- I/O helpers -----
inline void fwrite_exact(const void* p, size_t n, FILE* f = stdout) {
  size_t w = std::fwrite(p, 1, n, f);
  if (w != n) die("error: short write");
}
inline void fread_exact(void* p, size_t n, FILE* f = stdin) {
  size_t r = std::fread(p, 1, n, f);
  if (r != n) die("error: short read");
}

// Modern name we used in new main:
inline size_t read_chunk_into_ptr(uint8_t* dst, size_t max_n, FILE* f = stdin) {
  size_t total = 0;
  while (total < max_n) {
    size_t got = std::fread(dst + total, 1, max_n - total, f);
    total += got;
    if (got == 0) break; // EOF or short read
  }
  return total;
}

// Back-compat shim for older calls inside mgpu.cpp:
inline size_t read_chunk(void* dst, size_t max_n) {
  return read_chunk_into_ptr(static_cast<uint8_t*>(dst), max_n);
}

// ----- Progress bar rendering -----
inline void render_progress_bar(double percentage, uint64_t current, uint64_t total, double ratio, double speed_in, double speed_out, uint64_t chunks, bool enable_checksum = false) {
  const int bar_width = 20;
  int filled_width = static_cast<int>(percentage * bar_width);

  // Create progress bar
  std::string bar = "[";
  for (int i = 0; i < bar_width; ++i) {
    if (i < filled_width) {
      bar += "█";  // Full block character
    } else if (i == filled_width && percentage < 1.0) {
      bar += ">";  // Arrow for partial completion
    } else {
      bar += " ";
    }
  }
  bar += "]";

  // Format numbers
  double current_mb = current / (1024.0 * 1024.0);
  double total_mb = total / (1024.0 * 1024.0);

  // Determine if we're compressing or decompressing based on ratio
  bool is_compressing = (ratio > 1.0);  // ratio > 1 means compression (input > output)
  std::string mode = is_compressing ? "compress" : "decompress";

  std::string checksum_info = enable_checksum ? " [checksum]" : "";
  std::fprintf(stderr, "\r%s: %.1f%% %s %.1f/%.1f MB (%.2fx) %.1f/%.1f MB/s %lu chunks%s",
              mode.c_str(),
              percentage * 100.0,
              bar.c_str(),
              current_mb,
              total_mb,
              ratio,
              speed_in,
              speed_out,
              chunks,
              checksum_info.c_str());
  std::fflush(stderr);
}

} // namespace nvcz
