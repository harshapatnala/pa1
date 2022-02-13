# Location of the CUDA Toolkit
NVCC := $(CUDA_PATH)/bin/nvcc
CCFLAGS := -O2
EXTRA_NVCCFLAGS := --cudart=shared

build: quamsimV1 quamsimV2

quamsimV1.o:quamsimV1.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV2.o:quamsimV2.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV1: quamsimV1.o
	$(NVCC) $(LDFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

quamsimV2: quamsimV2.o
	$(NVCC) $(LDFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./quamsimV1

clean:
	rm -f quamsimV1 *.o
	rm -f quamsimV2 *.o
