
#ARCH=-arch sm_30 #not sure if need later for CUDA

BIN=x.kCPU
COMP=gcc
OPTS=-O3 #$(ARCH)

all: $(BIN)

$(BIN): main.c kmeans.c kmeans.h
	$(COMP) $(OPTS) main.o kmeans.o -o $(BIN)
main.0: main.c kmeans.c kmeans.h
	$(COMP) $(OPTS) -c main.c -o main.o

kmeans.o: kmeans.c kmeans.h
	$(COMP) $(OPTS) -c kmeans.c -o kmeans.o

clean:
	rm -rf kmeans.o main.o $(BIN)
