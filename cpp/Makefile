SHELL := /bin/bash
CPP := g++
CFLAGS := -O3 -march=native -std=c++11 -Wall
LDFLAGS := -ltensorflow `pkg-config --libs --cflags opencv`
ROSFLAGS := `pkg-config --libs --cflags rosbag` `pkg-config --libs --cflags sensor_msgs` `pkg-config --libs --cflags cv_bridge`

all:
	$(CPP) src/main.cpp src/unet.cpp $(CFLAGS) $(LDFLAGS) -Isrc -o main
bag:
	$(CPP) src/add_to_bag.cpp src/unet.cpp $(CFLAGS) $(LDFLAGS) $(ROSFLAGS) -Isrc -o add_to_bag
clean:
	rm main
