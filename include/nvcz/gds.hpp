#pragma once
#include <cuda_runtime.h>
#include <cufile.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace nvcz {

// GDS/cuFile integration for zero-copy I/O
class GDSFile {
private:
    CUfileHandle_t handle_;
    bool is_open_ = false;
    int fd_ = -1;
    
public:
    GDSFile() = default;
    ~GDSFile() { close(); }
    
    // Non-copyable
    GDSFile(const GDSFile&) = delete;
    GDSFile& operator=(const GDSFile&) = delete;
    
    // Movable
    GDSFile(GDSFile&& other) noexcept : handle_(other.handle_), is_open_(other.is_open_) {
        other.is_open_ = false;
    }
    
    GDSFile& operator=(GDSFile&& other) noexcept {
        if (this != &other) {
            close();
            handle_ = other.handle_;
            is_open_ = other.is_open_;
            other.is_open_ = false;
        }
        return *this;
    }
    
    bool open(const char* filename, int flags);
    bool open_fd(int fd);
    void close();
    
    // Direct GPU memory I/O
    ssize_t read_to_gpu(void* d_ptr, size_t size, off_t offset);
    ssize_t write_from_gpu(const void* d_ptr, size_t size, off_t offset);
    
    bool is_open() const { return is_open_; }
};

// Initialize/cleanup GDS subsystem
bool gds_init();
void gds_cleanup();

} // namespace nvcz
