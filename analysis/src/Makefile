

BIN=x.kCPU
COMP=gcc
OPTS=-O3 -std=c11

$(BIN): main.o kmeans.o
	$(COMP) $(OPTS) -o $(BIN) main.o kmeans.o 
	
main.o: main.c kmeans.c kmeans.h
	$(COMP) $(OPTS) -c main.c 

kmeans.o: kmeans.c kmeans.h
	$(COMP) $(OPTS) -c kmeans.c 

clean:
	rm -rf kmeans.o main.o $(BIN)
