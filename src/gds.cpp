#include "nvcz/gds.hpp"
#include "nvcz/util.hpp"
#include <fcntl.h>
#include <unistd.h>

namespace nvcz {

static bool gds_initialized = false;

bool gds_init() {
    if (gds_initialized) return true;
    
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "Warning: cuFileDriverOpen failed (%d), GDS unavailable\n", status.err);
        return false;
    }
    
    gds_initialized = true;
    return true;
}

void gds_cleanup() {
    if (!gds_initialized) return;
    
    CUfileError_t status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "Warning: cuFileDriverClose failed (%d)\n", status.err);
    }
    
    gds_initialized = false;
}

bool GDSFile::open(const char* filename, int flags) {
    if (is_open_) close();
    
    if (!gds_initialized && !gds_init()) {
        return false; // Fall back to regular I/O
    }
    
    // Open regular file descriptor first
    fd_ = ::open(filename, flags, 0644);
    if (fd_ < 0) {
        return false;
    }
    
    // Register with cuFile
    CUfileDescr_t desc = {};
    desc.handle.fd = fd_;
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    
    CUfileError_t status = cuFileHandleRegister(&handle_, &desc);
    if (status.err != CU_FILE_SUCCESS) {
        ::close(fd_);
        fd_ = -1;
        std::fprintf(stderr, "Warning: cuFileHandleRegister failed (%d), falling back to regular I/O\n", status.err);
        return false;
    }
    
    is_open_ = true;
    return true;
}

bool GDSFile::open_fd(int fd) {
    if (is_open_) close();
    if (!gds_initialized && !gds_init()) {
        return false;
    }
    fd_ = fd;
    CUfileDescr_t desc = {};
    desc.handle.fd = fd_;
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileError_t status = cuFileHandleRegister(&handle_, &desc);
    if (status.err != CU_FILE_SUCCESS) {
        fd_ = -1;
        return false;
    }
    is_open_ = true;
    return true;
}

void GDSFile::close() {
    if (!is_open_) return;
    
    cuFileHandleDeregister(handle_);
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    
    is_open_ = false;
}

ssize_t GDSFile::read_to_gpu(void* d_ptr, size_t size, off_t offset) {
    if (!is_open_) return -1;
    
    return cuFileRead(handle_, d_ptr, size, offset, 0);
}

ssize_t GDSFile::write_from_gpu(const void* d_ptr, size_t size, off_t offset) {
    if (!is_open_) return -1;
    
    return cuFileWrite(handle_, d_ptr, size, offset, 0);
}

} // namespace nvcz
