#pragma once
#include <cstdint>
#include <vector>

namespace nvcz {

struct AutoTune {
  uint32_t chunk_mb = 32;
  int streams = 2;
};

struct PcieInfo { int gen=0, width=0; };

bool nvml_query_pcie(PcieInfo& out);
AutoTune pick_tuning(bool verbose);

// NEW (just decls; impls for ids/policy go in mgpu.cpp or autotune.cpp as you prefer)
std::vector<int> discover_gpus_ids();

} // namespace nvcz
