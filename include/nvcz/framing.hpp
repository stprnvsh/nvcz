#pragma once
#include <cstdint>
#include <cstring>

namespace nvcz {

// Simple stream framing for nvcz
// [Header], then repeated blocks of: [u64 raw_len][u64 comp_len][comp payload], then [0][0] trailer.

static constexpr char MAGIC[5] = {'N','V','C','Z','\0'};

struct Header {
  char     magic[5];   // "NVCZ\0"
  uint32_t version;    // 1
  uint8_t  algo;       // nvcz::Algo as uint8_t
  uint32_t chunk_mb;   // producer chunk size in MiB (consumer honors this)
};

} // namespace nvcz
