# Miori — Metal + C++20, no CMake, no Python.
CXX      ?= clang++
CXXFLAGS ?= -std=c++20 -O2 -g -Wall -Wextra -Wno-unused-parameter \
            -Iconfig -Ithird_party/metal-cpp -Isrc
LDFLAGS  ?= -framework Metal -framework Foundation -framework QuartzCore
METAL    ?= xcrun -sdk macosx metal -std=metal4.1 -O3 -fno-fast-math -Iconfig -Ishaders
METALLIB ?= xcrun -sdk macosx metallib

BUILD   := build
SHADERS := $(wildcard shaders/*.metal)
AIRS    := $(patsubst shaders/%.metal,$(BUILD)/%.air,$(SHADERS))
SRCS    := $(wildcard src/*.cpp)
OBJS    := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS))

.PHONY: all clean shaders
all: $(BUILD)/miori $(BUILD)/miori.metallib

$(BUILD)/miori: $(OBJS) | $(BUILD)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

$(BUILD)/%.o: src/%.cpp $(wildcard src/*.h) config/m6.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.air: shaders/%.metal $(wildcard shaders/*.h) config/m6.h | $(BUILD)
	$(METAL) -c $< -o $@

$(BUILD)/miori.metallib: $(AIRS)
	$(METALLIB) $(AIRS) -o $@

shaders: $(BUILD)/miori.metallib

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
