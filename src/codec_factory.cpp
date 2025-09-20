#include "nvcz/codec.hpp"

namespace nvcz {

// Global configuration for nvCOMP chunk size (default 64KB)
size_t g_nvcomp_chunk_size_kb = 64;  // Default 64KB

// forward decls provided by each codec TU
std::unique_ptr<Codec> make_codec_lz4(size_t chunk_size_kb);
std::unique_ptr<Codec> make_codec_gdeflate(size_t chunk_size_kb);
std::unique_ptr<Codec> make_codec_snappy(size_t chunk_size_kb);
std::unique_ptr<Codec> make_codec_zstd(size_t chunk_size_kb);

std::unique_ptr<Codec> make_codec(Algo a) {
  return make_codec(a, g_nvcomp_chunk_size_kb);
}

std::unique_ptr<Codec> make_codec(Algo a, size_t nvcomp_chunk_size_kb) {
  switch (a) {
    case Algo::LZ4:       return make_codec_lz4(nvcomp_chunk_size_kb);
    case Algo::GDEFLATE:  return make_codec_gdeflate(nvcomp_chunk_size_kb);
    case Algo::SNAPPY:    return make_codec_snappy(nvcomp_chunk_size_kb);
    case Algo::ZSTD:      return make_codec_zstd(nvcomp_chunk_size_kb);
    default: return {};
  }
}

} // namespace nvcz
