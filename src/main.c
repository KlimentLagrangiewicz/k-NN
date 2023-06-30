#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "help.h"
#include "knn.h"

long long TimeValue = 0;

unsigned long long time_RDTSC() {
    union ticks {
        unsigned long long tx;
        struct dblword {
            long tl, th;
        } dw;
    } t;
    __asm__ ("rdtsc\n"
      : "=a" (t.dw.tl), "=d"(t.dw.th)
      );
    return t.tx;
}

void timeStart() { TimeValue = time_RDTSC(); }

long long timeStop() { return time_RDTSC() - TimeValue; }

int main(int argc, char **argv) {
    if (argc < 8) {
        puts("Not enough parameters...");
        exit(1);
    }
    const int n1 = atoi(argv[1]), m = atoi(argv[2]), n2 = atoi(argv[3]), k = atoi(argv[4]);
    if ((n1 < 0) || (n2 < 0) || (m < 0) || (k < 0) || (k > n1)) {
        puts("Value of parametes is uncorrect...");
        exit(1);
    }
    double *xTr = (double*)malloc(n1 * m * sizeof(double));
    double *xTs = (double*)malloc(n2 * m * sizeof(double));
    int *yTr = (int*)malloc(n1 * sizeof(int));
    int *res = (int*)malloc(n2 * sizeof(int));
    fscanfTrainData(argv[5], xTr, yTr, n1, m);
    fscanfTestData(argv[6], xTs, n2 * m);
    clock_t cl = clock();
    timeStart();
    kNN(xTr, yTr, xTs, res, n1, m, n2, k);
    long long t = timeStop();
    cl = clock() - cl;
    if (t < 0) {
        printf("Time for k-NN classification = %lf s.\n", (double)cl);
    } else {
        printf("Time for k-NN classification = %lld CPU clocks;\n", t);
    }
    if (argc > 8) {
        int *ideal = (int*)malloc(n2 * sizeof(int));
        fscanfSplitting(argv[8], ideal, n2);
        const double a = getAccuracy(res, ideal, n2);
        printf("Accuracy of classification = %.5lf;\n", a);
        if (argc > 9) {
            fprintfFullResult(argv[9], res, n1, m, n2, k, a);
        }
        fprintfShortFullyResult(argv[7], n1, m, n2, k, a);
        free(ideal);
    } else {
        fprintfResult(argv[7], res, n1, m, n2, k);
    }
    free(xTr);
    free(xTs);
    free(yTr);
    free(res);
    return 0;
}
