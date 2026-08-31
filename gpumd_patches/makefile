###########################################################
# Note: 
# 1) You can modify -arch=sm_60 according to 
#    your GPU architecture. For compute capability < 6.0,
#    need to add -DUSE_KEPLER to CFLAGS.
# 2) For Windows systems, if you get errors like 
#    c1xx : fatal error C1083: Cannot open source file: ...
#    You can consider removing -Xcompiler "/wd 4819"
# 3) Add -DUSE_PLUMED to CFLAGS when use the PLUMED plugin
#    and remove it otherwise.
###########################################################


###########################################################
# some flags
###########################################################
CC = nvcc
CUDA_ARCH=-arch=sm_60
WARN_FLAGS = -Wall -Wextra
WARN_FLAGS += -Wno-sign-compare -Wno-unused-parameter
WARN_FLAGS += -Wnull-dereference -Wstringop-overflow -Wstringop-truncation -Wrestrict
WARN_FLAGS += -Wduplicated-cond -Wduplicated-branches -Wlogical-op
WARN_FLAGS += -Wsequence-point -Wswitch-default
WARN_FLAGS += -Wcast-align -Wcast-qual -Wvla
WARN_FLAGS += -Wformat -Wformat-security -Wformat-overflow -Wformat-truncation
ifdef OS # For Windows with the cl.exe compiler
CFLAGS = -O3 $(CUDA_ARCH)
else # For linux
# -fPIC is required so that the nvcc-compiled objects can be linked into the
# shared Python extension module (gpumd.so) by the pygpumd target.
CFLAGS = -std=c++14 -O3 $(CUDA_ARCH) -Xcompiler="$(WARN_FLAGS)" -Xcompiler -fPIC
endif
INC = -I./
LDFLAGS = 
LIBS = -lcublas -lcusolver -lcufft

# ---- Python / pybind11 configuration (for pygpumd target) ----
PYGPUMD_SRC = main_gpumd/gpumd_python.cpp
PYBIND11_INC  := $(shell python3 -m pybind11 --includes 2>/dev/null)
PYTHON_CFLAGS := $(shell python3-config --cflags 2>/dev/null)
PYTHON_LDFLAGS:= $(shell python3-config --ldflags --embed 2>/dev/null)
# If python3-config fails, fall back to simpler flags
ifeq ($(PYTHON_CFLAGS),)
  PYTHON_CFLAGS = -I/usr/include/python3
  PYTHON_LDFLAGS = -lpython3
endif
# CUDA include path: needed because run.cuh transitively includes
# headers that require curand_kernel.h.
ifneq ($(wildcard /usr/local/cuda/include/curand_kernel.h),)
  CUDA_INC := /usr/local/cuda/include
else
  CUDA_INC ?= /usr/local/cuda/targets/x86_64-linux/include
endif
# CUDA library path: needed for linking cublas, cusolver, cufft, cudart
CUDA_LIB := $(shell \
  pkg-config --libs cuda 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($$i ~ /^-L/) print substr($$i,3)}' | head -1 || \
  echo '/usr/local/cuda/targets/x86_64-linux/lib')
ifeq ($(wildcard $(CUDA_LIB)/libcusolver.so),)
  ifneq ($(wildcard /usr/local/cuda/lib64/libcusolver.so),)
    CUDA_LIB := /usr/local/cuda/lib64
  else
    CUDA_LIB ?= /usr/local/cuda/targets/x86_64-linux/lib
  endif
endif
# --------------------------------------------------------------


###########################################################
# source files
###########################################################
SOURCES_GPUMD =                   \
	$(wildcard main_gpumd/*.cu)   \
	$(wildcard minimize/*.cu)     \
	$(wildcard phonon/*.cu)       \
	$(wildcard integrate/*.cu)    \
	$(wildcard mc/*.cu)           \
	$(wildcard force/*.cu)        \
	$(wildcard measure/*.cu)      \
	$(wildcard model/*.cu)        \
	$(wildcard utilities/*.cu)
SOURCES_NEP =                     \
	$(wildcard main_nep/*.cu)     \
	$(wildcard utilities/*.cu)
SOURCES_GNEP =                     \
	$(wildcard main_gnep/*.cu)     \
	$(wildcard utilities/*.cu)


###########################################################
# object files
###########################################################
ifdef OS # For Windows with the cl.exe compiler
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.obj)
OBJ_NEP = $(SOURCES_NEP:.cu=.obj)
OBJ_GNEP = $(SOURCES_GNEP:.cu=.obj)
else
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.o)
OBJ_NEP = $(SOURCES_NEP:.cu=.o)
OBJ_GNEP = $(SOURCES_GNEP:.cu=.o)
endif


###########################################################
# headers
###########################################################
HEADERS =                         \
	$(wildcard utilities/*.cuh)   \
	$(wildcard main_gpumd/*.cuh)  \
	$(wildcard integrate/*.cuh)   \
	$(wildcard mc/*.cuh)          \
	$(wildcard minimize/*.cuh)    \
	$(wildcard force/*.cuh)       \
	$(wildcard measure/*.cuh)     \
	$(wildcard model/*.cuh)       \
	$(wildcard phonon/*.cuh)      \
	$(wildcard main_nep/*.cuh)    \
	$(wildcard main_gnep/*.cuh)


###########################################################
# executables
###########################################################
all: gpumd nep 
gpumd: $(OBJ_GPUMD)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The gpumd executable is successfully compiled!
	@echo =================================================
nep: $(OBJ_NEP)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The nep executable is successfully compiled!
	@echo =================================================
gnep: $(OBJ_GNEP)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The gnep executable is successfully compiled!
	@echo =================================================


############################################################
#pygpumd: Python extension module for PySAGES integration 
############################################################
ifdef OS
PYGPUMD_OBJ = $(PYGPUMD_SRC:.cpp=.obj)
else
PYGPUMD_OBJ = $(PYGPUMD_SRC:.cpp=.o)
endif

PY_OBJ_GPUMD = $(filter-out %/main.o %/main.obj,$(OBJ_GPUMD)) $(PYGPUMD_OBJ)

pygpumd: CFLAGS += -DUSE_PYSAGES
pygpumd: $(PY_OBJ_GPUMD)
ifdef OS
	@echo Windows Python module build not yet implemented in makefile.
else
	g++ -shared -fPIC $(LDFLAGS) $^ -o gpumd.so \
	  -L$(CUDA_LIB) -Wl,-rpath,$(CUDA_LIB) \
	  $(LIBS) -lcudart $(PYTHON_LDFLAGS)
	@echo =================================================
	@echo The gpumd Python module is successfully compiled!
	@echo =================================================
endif


###########################################################
# rules for building object files
###########################################################
ifdef OS # for Windows
integrate/%.obj: integrate/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
mc/%.obj: mc/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
minimize/%.obj: minimize/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
force/%.obj: force/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
measure/%.obj: measure/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gpumd/%.obj: main_gpumd/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
utilities/%.obj: utilities/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
model/%.obj: model/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
phonon/%.obj: phonon/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_nep/%.obj: main_nep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gnep/%.obj: main_gnep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
else # for Linux
integrate/%.o: integrate/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
mc/%.o: mc/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
minimize/%.o: minimize/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
force/%.o: force/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
measure/%.o: measure/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gpumd/%.o: main_gpumd/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
utilities/%.o: utilities/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
model/%.o: model/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
phonon/%.o: phonon/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_nep/%.o: main_nep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gnep/%.o: main_gnep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gpumd/%.o: main_gpumd/%.cpp $(HEADERS)
	g++ -std=c++14 -O3 -fPIC -DUSE_PYSAGES $(PYBIND11_INC) $(PYTHON_CFLAGS) -I$(CUDA_INC) $(INC) -c $< -o $@
endif


###########################################################
# clean up
###########################################################
clean:
ifdef OS
	del /s *.obj *.exp *.lib *.exe
else
	rm -f */*.o main_gpumd/*.o gpumd nep gnep gpumd.so
endif

