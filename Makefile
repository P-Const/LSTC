# Makefile

SOURCES := lstc.cpp main.cpp

override CXXFLAGS += -std=c++1y -O2 -fopenmp
override CPPFLAGS += -I include
override LDFLAGS += -fopenmp

vpath %.hpp include
vpath %.cpp src
vpath %.o output

.PHONY: clean

lstc: $(SOURCES:%.cpp=output/%.o)
	$(CXX) $(LDFLAGS) -o $@ $^

output/%.o: %.cpp lstc.hpp | out
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

out:
	mkdir -p output	

clean:
	rm -rf output/*.o