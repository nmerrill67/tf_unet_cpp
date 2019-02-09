SHELL := /bin/bash
CPP := g++
CFLAGS := -O3 -march=native -std=c++11 -fPIC -Wall
LDFLAGS := -ltensorflow `pkg-config --libs --cflags opencv`

all:
	$(CPP) $(CFLAGS) $(LDFLAGS) main.cpp unet.cpp -I./ -o main

clean:
	rm main
