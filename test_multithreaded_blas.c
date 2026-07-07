#include <stdio.h>

int main() {
#ifdef MULTITHREADED_BLAS
    printf("MULTITHREADED_BLAS is DEFINED\n");
    return 0;
#else
    printf("MULTITHREADED_BLAS is NOT DEFINED\n");
    return 1;
#endif
}
