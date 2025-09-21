CXX ?= g++

# --- locate CUDA ---
# allow override: make CUDA_HOME=/path/to/cuda
CUDA_HOME ?= /opt/cuda
CUDA_INC  ?= $(CUDA_HOME)/include
CUDA_LIB  ?= $(CUDA_HOME)/lib64

# --- locate nvCOMP (Arch usually puts headers in /usr/include/nvcomp, lib in /usr/lib) ---
NVCOMP_INC ?= /usr/include
NVCOMP_LIB ?= /usr/lib

CXXFLAGS := -O2 -std=c++17 -Iinclude -I$(CUDA_INC) -I$(NVCOMP_INC)
LDFLAGS  := -L$(CUDA_LIB) -L$(NVCOMP_LIB) -lnvcomp -lcudart -lcufile -ldl

SRC := src/util.cpp src/autotune.cpp \
       src/codec_lz4.cpp src/codec_gdeflate.cpp src/codec_snappy.cpp src/codec_zstd.cpp \
       src/codec_factory.cpp src/mgpu.cpp src/gds.cpp src/main.cpp

BIN := nvcz

# Installation paths
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin

all: sanity $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

sanity:
	@test -f $(CUDA_INC)/cuda_runtime.h || { echo ">>> cuda_runtime.h not found in $(CUDA_INC). Set CUDA_HOME=/opt/cuda (or your path)"; exit 1; }
	@echo "Using CUDA includes: $(CUDA_INC)"
	@echo "Using CUDA libs:     $(CUDA_LIB)"
	@echo "Using nvCOMP inc:    $(NVCOMP_INC)"
	@echo "Using nvCOMP libs:   $(NVCOMP_LIB)"

install: $(BIN)
	@mkdir -p $(BINDIR)
	@cp $(BIN) $(BINDIR)/$(BIN)
	@chmod +x $(BINDIR)/$(BIN)
	@echo "nvcz installed to $(BINDIR)/$(BIN)"

uninstall:
	@rm -f $(BINDIR)/$(BIN)
	@echo "nvcz uninstalled from $(BINDIR)/$(BIN)"

help:
	@echo "nvcz - High-Performance GPU-Accelerated Compression"
	@echo ""
	@echo "Available targets:"
	@echo "  all       - Build nvcz (default)"
	@echo "  install   - Install nvcz to $(BINDIR)"
	@echo "  uninstall - Remove nvcz from $(BINDIR)"
	@echo "  clean     - Remove built binary"
	@echo "  help      - Show this help"
	@echo ""
	@echo "Installation paths can be customized:"
	@echo "  make PREFIX=/usr install          # Install to /usr/bin"
	@echo "  make BINDIR=/opt/bin install      # Install to /opt/bin"

clean:
	rm -f $(BIN)
