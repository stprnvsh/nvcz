#include "nvcz/autotune.hpp"
#include "nvcz/util.hpp"
#include <dlfcn.h>
#include <cstdio>
#include <algorithm>
#include "nvcz/codec.hpp" 

namespace nvcz {

bool nvml_query_pcie(PcieInfo& pi) {
  using nvmlInit_t = int(*)();
  using nvmlShutdown_t = int(*)();
  using nvmlDeviceGetHandleByIndex_t = int(*)(unsigned,int*);
  using nvmlDeviceGetMaxPcieLinkWidth_t = int(*)(int*, int*);
  using nvmlDeviceGetMaxPcieLinkGeneration_t = int(*)(int*, int*);
  void* h = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  if (!h) return false;
  auto nvmlInit = (nvmlInit_t)dlsym(h, "nvmlInit_v2");
  auto nvmlShutdown = (nvmlShutdown_t)dlsym(h, "nvmlShutdown");
  auto getHandle = (nvmlDeviceGetHandleByIndex_t)dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
  auto getWidth  = (nvmlDeviceGetMaxPcieLinkWidth_t)dlsym(h, "nvmlDeviceGetMaxPcieLinkWidth");
  auto getGen    = (nvmlDeviceGetMaxPcieLinkGeneration_t)dlsym(h, "nvmlDeviceGetMaxPcieLinkGeneration");
  if (!nvmlInit || !nvmlShutdown || !getHandle || !getWidth || !getGen) { dlclose(h); return false; }
  if (nvmlInit() != 0) { dlclose(h); return false; }
  int dev=0, hdl=0;
  if (getHandle(dev, &hdl)!=0) { nvmlShutdown(); dlclose(h); return false; }
  int w=0,g=0;
  getWidth(&hdl, &w);
  getGen(&hdl, &g);
  nvmlShutdown(); dlclose(h);
  if (w>0 && g>0) { pi.width=w; pi.gen=g; return true; }
  return false;
}

AutoTune pick_tuning(bool verbose) {
  cudaDeviceProp p{};
  int dev=0; cuda_ck(cudaGetDevice(&dev), "get device");
  cuda_ck(cudaGetDeviceProperties(&p, dev), "get props");
  double mem_bw_gbps =
      (double)p.memoryClockRate * 1000.0 /*kHz→Hz*/
    * (double)p.memoryBusWidth / 8.0     /*bits→bytes*/
    * 2.0 /*DDR*/ / 1e9;

  PcieInfo pci{}; bool have_pci = nvml_query_pcie(pci);
  AutoTune t{};
  if (p.totalGlobalMem < (size_t)8ull<<30) t.chunk_mb = 16;
  else if (p.totalGlobalMem < (size_t)24ull<<30) t.chunk_mb = 32;
  else t.chunk_mb = 64;

  int ce = p.asyncEngineCount > 0 ? p.asyncEngineCount : 1;
  t.streams = std::min(4, std::max(2, ce+1));

  if (have_pci && (pci.gen <= 3 || pci.width <= 8)) {
    if (t.chunk_mb >= 64) t.chunk_mb = 32;
    t.streams = std::max(2, t.streams-1);
  }
  if (mem_bw_gbps < 300.0) t.streams = std::max(2, t.streams-1);

  if (verbose) {
    std::fprintf(stderr,
      "[auto] VRAM=%.1f GiB, memBW≈%.0f GB/s, asyncCE=%d, PCIe Gen%d x%d -> chunk=%uMB, streams=%d\n",
      (double)p.totalGlobalMem/(1<<30), mem_bw_gbps, p.asyncEngineCount,
      pci.gen, pci.width, t.chunk_mb, t.streams);
  }
  return t;
}

} // namespace nvcz
