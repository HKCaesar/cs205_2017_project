
#ARCH=-arch sm_30 #not sure if need later for CUDA

BIN=x.kCPU
COMP=gcc
OPTS=-O3 #$(ARCH)

all: 
	$(BIN)

$(BIN): main.o kmeans.o
	$(COMP) $(OPTS) -o $(BIN) main.o kmeans.o 
	
main.o: main.c kmeans.c kmeans.h
	$(COMP) $(OPTS) -c main.c -o main.o

kmeans.o: kmeans.c kmeans.h
	$(COMP) $(OPTS) -c kmeans.c -o kmeans.o

clean:
	rm -rf kmeans.o main.o $(BIN)
