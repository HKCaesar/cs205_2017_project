
#ARCH=-arch sm_30 #not sure if need later for CUDA

BIN=x.kCPU
COMP=gcc
OPTS=-O3 #$(ARCH)

all: $(BIN)

$(BIN): main.c kmeans.o
	$(COMP) $(OPTS)  -o $(BIN) main.c kmeans.o

kmeans.o: kmeans.c
	$(COMP) $(OPTS) -c kmeans.c -o kmeans.o


clean:
	rm -rf kmeans.o $(BIN)
