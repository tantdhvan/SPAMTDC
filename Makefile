CXX      = g++
CPPFLAGS = -std=c++17 -O2 -Wall -Isrc

OMPFLAGS = -fopenmp

LDLIBS   = -lpthread

ALG_HEADERS = \
    src/algs/potentialswap.h \
    src/algs/result.h \
    src/algs/stream_utils.h \
    src/algs/streaming_literature.h

CORE_HEADERS = \
    src/kfunctions.h \
    src/kfunctions_impl.h \
    src/mygraph.h \
    src/objectvalue/kic.h \
    src/objectvalue/lt.h \
    src/objectvalue/suitability.h \
    src/prefix_stream.h

.PHONY: all clean debug

all: kic lt preprockic

kic: src/main.cpp $(ALG_HEADERS) $(CORE_HEADERS)
	$(CXX) src/main.cpp -o kic \
	    $(CPPFLAGS) $(OMPFLAGS) -DKFUNC_KIC $(LDLIBS)

lt: src/main.cpp $(ALG_HEADERS) $(CORE_HEADERS)
	$(CXX) src/main.cpp -o lt \
	    $(CPPFLAGS) $(OMPFLAGS) -DKFUNC_LT $(LDLIBS)

preprockic: src/data/preprocess_kic.cpp
	$(CXX) src/data/preprocess_kic.cpp -o preprockic \
	    $(CPPFLAGS) $(LDLIBS)

debug: src/main.cpp
	$(CXX) src/main.cpp -o kic_debug \
	    -std=c++17 -O0 -g -ggdb3 -Wall -Isrc $(OMPFLAGS) -DKFUNC_KIC $(LDLIBS)

clean:
	rm -f kic lt preprockic kic_debug *.exe
