hello:hello.o
	nvcc -o hello hello.o

hello.o:hello.py
	nvcc -c hello.py

clean:
	rm *.o *.gch hello
