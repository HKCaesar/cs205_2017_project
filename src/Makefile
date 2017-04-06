

#ARCH=-arch sm_30 #not sure if need later for CUDA

BIN=x.kCPU
COMP=gcc
OPTS=-O3 #$(ARCH)

$(BIN): kmeans.o
	$(COMP) $(OPTS) -o $(BIN) kmeans.o

kmeans.o: kmeans.c
	$(COMP) $(OPTS) -c kmeans.c 

clean:
	rm -rf kmeans.o $(BIN)
