SHELL := /bin/bash
CPP := g++
CFLAGS := -O3 -march=native -std=c++11 -fPIC -Wall
LDFLAGS := -ltensorflow `pkg-config --libs --cflags opencv`

all:
	$(CPP) main.cpp unet.cpp $(CFLAGS) $(LDFLAGS) -I./ -o main

clean:
	rm main
