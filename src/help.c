#include "help.h"

void fscanfTrainData(const char *fn, double *x, int *y, const int n, const int m) {
    FILE *fl = fopen(fn, "r");
    if (fl == NULL) {
        printf("Error in opening file %s with train data set...\n", fn);
        exit(1);
    }
    int i, j;
    for (i = 0; i < n && !feof(fl); i++) {
        for (j = i * m; (j < i * m + m) && !feof(fl); j++) {
            fscanf(fl, "%lf", &x[j]);
        }
        if (!feof(fl))
            fscanf(fl, "%d", &y[i]);
    }
    fclose(fl);
}

void fscanfTestData(const char *fn, double *x, const int n) {
    FILE *fl = fopen(fn, "r");
    if (fl == NULL) {
        printf("Error in opening file %s with test data set...\n", fn);
        exit(1);
    }
    int i;
    for (i = 0; i < n && !feof(fl); i++) {
        fscanf(fl, "%lf", &x[i]);
    }
    fclose(fl);
}

void fscanfSplitting(const char *fn, int *y, const int n) {
    FILE *fl = fopen(fn, "r");
    if (fl == NULL) {
        printf("Error in opening file %s with ideal splitting...\n", fn);
        exit(1);
    }
    int i = 0;
    while (i < n && !feof(fl)) {
        fscanf(fl, "%d", &y[i]);
        i++;
    }
    fclose(fl);
}

void fprintfResult(const char *fn, const int *y, const int n1, const int m, const int n2, const int k) {
    FILE *fl = fopen(fn, "a");
    if (fl == NULL) {
        printf("Error in opening file %s for printing results...\n", fn);
        exit(1);
    }
    fprintf(fl, "Result of k-NN clussification...\nN = %d, M = %d, Ntest = %d, K = %d;\n", n1, m, n2, k);
    int i = 0;
    while (i < n2) {
        fprintf(fl, "Object[ %d ] = %d;\n", i, y[i]);
        i++;
    }
    fprintf(fl, "\n");
    fclose(fl);
}

void fprintfShortFullyResult(const char *fn, const int n1, const int m, const int n2, const int k, const double a) {
    FILE *fl = fopen(fn, "a");
    if (fl == NULL) {
        printf("Error in opening file %s for printing results...\n", fn);
        exit(1);
    }
    fprintf(fl, "Result of k-NN clussification...\nN = %d, M = %d, Ntest = %d, K = %d;\nAccuracy of classification = %.5lf;\n\n", n1, m, n2, k, a);
    fclose(fl);
}

double getAccuracy(const int *x, const int *y, const int n) {
    int i = 0, j = 0;
    while (i < n) {
        if (x[i] == y[i]) {
            j++;
        }
        i++;
    }
    return (double)j/ (double)n;
}

void fprintfFullResult(const char *fn, const int *y, const int n1, const int m, const int n2, const int k, const double a) {
    FILE *fl = fopen(fn, "a");
    if (fl == NULL) {
        printf("Error in opening file %s for printing results...\n", fn);
        exit(1);
    }
    fprintf(fl, "Result of k-NN clussification...\nN = %d, M = %d, Ntest = %d, K = %d;\nAccuracy of classification = %.5lf;\n", n1, m, n2, k, a);
    int i = 0;
    while (i < n2) {
        fprintf(fl, "Object[ %d ] = %d;\n", i, y[i]);
        i++;
    }
    fprintf(fl, "\n");
    fclose(fl);
}


