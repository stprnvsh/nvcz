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
LDFLAGS  := -L$(CUDA_LIB) -L$(NVCOMP_LIB) -lnvcomp -lcudart -ldl

# Core library sources (used by both CLI and library builds)
LIB_SRC := src/util.cpp src/autotune.cpp \
           src/codec_lz4.cpp src/codec_gdeflate.cpp src/codec_snappy.cpp src/codec_zstd.cpp \
           src/codec_factory.cpp src/mgpu.cpp src/nvcz.cpp

# CLI-specific sources
CLI_SRC := src/main.cpp

# Object files
LIB_OBJ := $(LIB_SRC:.cpp=.o)
CLI_OBJ := $(CLI_SRC:.cpp=.o)

# Output files
BIN := nvcz
LIB_SHARED := libnvcz.so
LIB_STATIC := libnvcz.a

# Installation paths
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
LIBDIR ?= $(PREFIX)/lib
INCDIR ?= $(PREFIX)/include

all: sanity cli

cli: sanity $(BIN)

lib: sanity $(LIB_SHARED) $(LIB_STATIC)

lib-shared: sanity $(LIB_SHARED)

lib-static: sanity $(LIB_STATIC)

all: sanity cli lib

# Library-specific flags
LIB_CXXFLAGS := $(CXXFLAGS) -fPIC -shared -fvisibility=hidden
LIB_LDFLAGS := $(LDFLAGS) -shared

# Build CLI executable
$(BIN): $(LIB_OBJ) $(CLI_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Build shared library
$(LIB_SHARED): $(LIB_OBJ)
	$(CXX) $(LIB_CXXFLAGS) $^ -o $@ $(LIB_LDFLAGS)

# Build static library
$(LIB_STATIC): $(LIB_OBJ)
	ar rcs $@ $^

# Pattern rule for object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

sanity:
	@test -f $(CUDA_INC)/cuda_runtime.h || { echo ">>> cuda_runtime.h not found in $(CUDA_INC). Set CUDA_HOME=/opt/cuda (or your path)"; exit 1; }
	@echo "Using CUDA includes: $(CUDA_INC)"
	@echo "Using CUDA libs:     $(CUDA_LIB)"
	@echo "Using nvCOMP inc:    $(NVCOMP_INC)"
	@echo "Using nvCOMP libs:   $(NVCOMP_LIB)"

install-cli: cli
	@mkdir -p $(BINDIR)
	@cp $(BIN) $(BINDIR)/$(BIN)
	@chmod +x $(BINDIR)/$(BIN)
	@echo "nvcz CLI installed to $(BINDIR)/$(BIN)"

install-lib: lib
	@mkdir -p $(LIBDIR)
	@cp $(LIB_SHARED) $(LIBDIR)/$(LIB_SHARED)
	@cp $(LIB_STATIC) $(LIBDIR)/$(LIB_STATIC)
	@mkdir -p $(INCDIR)
	@cp -r include/nvcz $(INCDIR)/
	@echo "nvcz library installed to $(LIBDIR)/"
	@echo "Headers installed to $(INCDIR)/nvcz/"

install: install-cli install-lib

uninstall-cli:
	@rm -f $(BINDIR)/$(BIN)
	@echo "nvcz CLI uninstalled from $(BINDIR)/$(BIN)"

uninstall-lib:
	@rm -f $(LIBDIR)/$(LIB_SHARED) $(LIBDIR)/$(LIB_STATIC)
	@rm -rf $(INCDIR)/nvcz
	@echo "nvcz library uninstalled from $(LIBDIR) and $(INCDIR)"

uninstall: uninstall-cli uninstall-lib

help:
	@echo "nvcz - High-Performance GPU-Accelerated Compression"
	@echo ""
	@echo "Available build targets:"
	@echo "  cli       - Build CLI executable (nvcz)"
	@echo "  lib       - Build shared and static libraries (libnvcz.so, libnvcz.a)"
	@echo "  lib-shared - Build shared library only (libnvcz.so)"
	@echo "  lib-static - Build static library only (libnvcz.a)"
	@echo "  all       - Build both CLI and libraries"
	@echo ""
	@echo "Installation targets:"
	@echo "  install-cli   - Install CLI executable"
	@echo "  install-lib   - Install libraries and headers"
	@echo "  install       - Install both CLI and libraries"
	@echo "  uninstall-cli - Remove CLI executable"
	@echo "  uninstall-lib - Remove libraries and headers"
	@echo "  uninstall     - Remove both CLI and libraries"
	@echo ""
	@echo "Other targets:"
	@echo "  clean     - Remove built files"
	@echo "  help      - Show this help"
	@echo ""
	@echo "Installation paths can be customized:"
	@echo "  make PREFIX=/usr install          # Install to /usr"
	@echo "  make BINDIR=/opt/bin install-cli  # Install CLI to /opt/bin"
	@echo "  make LIBDIR=/opt/lib install-lib  # Install library to /opt/lib"
	@echo "  make INCDIR=/opt/include install-lib # Install headers to /opt/include"

clean:
	rm -f $(BIN) $(LIB_SHARED) $(LIB_STATIC) $(LIB_OBJ) $(CLI_OBJ)
