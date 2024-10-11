#include "graph.h"

using namespace adf;

mygraph g;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(int argc, char ** argv) {
    g.init();
    g.run(10000);
    g.end();
    return 0;
};
#endif
