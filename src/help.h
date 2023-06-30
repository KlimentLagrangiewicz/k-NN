#ifndef HELP_H
#define HELP_H

#include <stdio.h>
#include <stdlib.h>

void fscanfTrainData(const char *fn, double *x, int *y, const int n, const int m);
void fscanfTestData(const char *fn, double *x, const int n);
void fscanfSplitting(const char *fn, int *y, const int n);
void fprintfResult(const char *fn, const int *y, const int n1, const int m, const int n2, const int k);
void fprintfShortFullyResult(const char *fn, const int n1, const int m, const int n2, const int k, const double a);
double getAccuracy(const int *x, const int *y, const int n);
void fprintfFullResult(const char *fn, const int *y, const int n1, const int m, const int n2, const int k, const double a);

#endif
