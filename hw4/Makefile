
CC = mpicc
CFLAGS = -O3 -march=native -fopenmp

CXX = mpicxx
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

C_FILES = $(wildcard *.c)
CPP_FILES = $(wildcard *.cpp)

C_TARGETS = $(basename $(C_FILES))
CPP_TARGETS = $(basename $(CPP_FILES))

all : $(C_TARGETS) $(CPP_TARGETS)

%:%.c *.h
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(C_TARGETS) $(CPP_TARGETS) *~

.PHONY: all, clean