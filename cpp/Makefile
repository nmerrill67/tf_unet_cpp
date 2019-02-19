SHELL := /bin/bash
CPP := g++
CFLAGS := -O3 -march=native -std=c++11 -Wall
LDFLAGS := -ltensorflow `pkg-config --libs --cflags opencv`
ROSFLAGS := `pkg-config --libs --cflags rosbag` `pkg-config --libs --cflags sensor_msgs` `pkg-config --libs --cflags cv_bridge`

all: obj bag main
main: obj
	$(CPP) src/main.cpp unet.o $(CFLAGS) $(LDFLAGS) -Isrc -o main
bag: obj
	$(CPP) src/add_to_bag.cpp unet.o $(CFLAGS) $(LDFLAGS) $(ROSFLAGS) -Isrc -o add_to_bag
obj:
	$(CPP) -c src/unet.cpp $(CFLAGS) $(LDFLAGS) -Isrc
clean:
	rm main
