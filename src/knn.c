#include "knn.h"

void autoscaling(double *x, double *y, const int n1, const int n2, const int m) {
    const int s1 = n1 * m, s2 = n2 * m;
    double sd, Ex, Exx;
    int i, j = 0;
    while (j < m) {
        i = j;
        Ex = Exx = 0;
        while (i < s1) {
            sd = x[i];
            Ex += sd;
            Exx += sd * sd;
            i += m;
        }
        i = j;
        while (i < s2) {
            sd = y[i];
            Ex += sd;
            Exx += sd * sd;
            i += m;
        }
        Exx /= n1 + n2;
        Ex /= n1 + n2;
        sd = sqrt(Exx - Ex * Ex);
        i = j;
        while (i < s1) {
            x[i] = (x[i] - Ex) / sd;
            i += m;
        }
        i = j;
        while (i < s2) {
            y[i] = (y[i] - Ex) / sd;
            i += m;
        }
        j++;
    }
}

double distEv(const double *x, const double *c, const int m) {
    double d, r = 0;
    int i = 0;
    while (i++ < m) {
        d = *(x++) - *(c++);
        r += d * d;
    }
    return r;
}

int getNumOfClass(const int *y, const int n) {
    int i, j, cur;
    char *v = (char*)malloc(n * sizeof(char));
    memset(v, 0, n * sizeof(char));
    for (i = 0; i < n; i++) {
        while ((v[i]) && (i < n)) i++;
        cur = y[i];
        for (j = i + 1; j < n; j++) {
            if (y[j] == cur)
                v[j] = 1;
        }
    }
    i = cur = 0;
    while (i < n) {
        if (v[i] == 0) cur++;
        i++;
    }
    free(v);
    return cur;
}

void InsertionSort(double *pr, int *cl, int count) {
    double key_pr;
    int i, key_cl, j;
    for (i = 1; i < count; i++) {
        key_pr = pr[i];
        key_cl = cl[i];
        j = i - 1;
        while (j >= 0 && pr[j] > key_pr) {
            pr[j + 1] = pr[j];
            cl[j + 1] = cl[j];
            j--;
        }
        pr[j + 1] = key_pr;
        cl[j + 1] = key_cl;
    }
}

void MaxHeapify(double *pr, int *cl, int heapSize, int index) {
    int left = (index + 1) * 2 - 1;
    int right = (index + 1) * 2;
    int largest = 0;
    if (left < heapSize && pr[left] > pr[index])
        largest = left;
    else
        largest = index;

    if (right < heapSize && pr[right] > pr[largest])
        largest = right;

    if (largest != index) {
        double temp_pr = pr[index];
        int temp_cl = cl[index];
        pr[index] = pr[largest];
        cl[index] = cl[largest];
        pr[largest] = temp_pr;
        cl[largest] = temp_cl;
        MaxHeapify(pr, cl, heapSize, largest);
    }
}

void HeapSort(double *pr, int *cl, int count) {
    int heapSize = count, p, i;
    for (p = (heapSize - 1) / 2; p >= 0; p--)
        MaxHeapify(pr, cl, heapSize, p);
    for (i = count - 1; i > 0; i--) {
        double temp_pr = pr[i];
        int temp_cl = cl[i];
        pr[i] = pr[0];
        cl[i] = cl[0];
        pr[0] = temp_pr;
        cl[0] = temp_cl;
        heapSize--;
        MaxHeapify(pr, cl, heapSize, 0);
    }
}

int Partition(double* pr, int *cl, int left, int right) {
    double pivot_pr = pr[right], temp_pr;
    int j, temp_cl, pivot_cl = cl[right], i = left;
    for (j = left; j < right; j++) {
        if (pr[j] <= pivot_pr) {
            temp_pr = pr[j];
            temp_cl = cl[j];
            pr[j] = pr[i];
            cl[j] = cl[i];
            pr[i] = temp_pr;
            cl[i] = temp_cl;
            i++;
        }
    }
    pr[right] = pr[i];
    cl[right] = cl[i];
    pr[i] = pivot_pr;
    cl[i] = pivot_cl;
    return i;
}

void QuickSortRecursive(double *pr, int *cl, int left, int right) {
    if (left < right) {
        int q = Partition(pr, cl, left, right);
        QuickSortRecursive(pr, cl, left, q - 1);
        QuickSortRecursive(pr, cl, q + 1, right);
    }
}

void IntroSort(double *pr, int *cl, int count) {
    int partitionSize = Partition(pr, cl, 0, count - 1);
    if (partitionSize < 16) {
        InsertionSort(pr, cl, count);
    }
    else if (partitionSize > (2 * log(count))) {
        HeapSort(pr, cl, count);
    } else {
        QuickSortRecursive(pr, cl, 0, count - 1);
    }
}

void kNN(double *xtr, const int *y, double *xts, int *res, const int n, const int m, const int n2, const int k) {
    int i, j, l, id, noc = getNumOfClass(y, n);
    autoscaling(xtr, xts, n, n2, m);
    double *d = (double*)malloc(n * sizeof(double));
    int *r = (int*)malloc(noc * sizeof(int));
    if ((log(n) / log(2)) < k) {
        int *cl = (int*)malloc(n * sizeof(int));
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n; j++) {
                d[j] = distEv(&xts[i * m], &xtr[j * m], m);
                cl[j] = y[j];
            }
            IntroSort(d, cl, n);
            memset(r, 0, noc * sizeof(int));
            for (j = 0; j < k; j++) {
                r[cl[j]]++;
            }
            id = 0;
            l = 0;
            for (j = 0; j < noc; j++) {
                if (r[j] > l) {
                    l = r[j];
                    id = j;
                }
            }
            res[i] = id;
        }
        free(cl);
    } else {
        double min_d;
        const size_t s1 = noc * sizeof(int), s2 = n * sizeof(char);
        char *v = (char*)malloc(s2);
        for (i = 0; i < n2; i++) {
            memset(v, 0, s2);
            memset(r, 0, s1);
            for (j = 0; j < n; j++) {
                d[j] = distEv(&xts[i * m], &xtr[j * m], m);
            }
            for (j = 0; j < k; j++) {
                l = 0;
                while (v[l] && l < n) l++;
                min_d = d[l];
                id = l;
                l++;
                for (; l < n; l++) {
                    if ((v[l] == 0) && (d[l] < min_d)) {
                        min_d = d[l];
                        id = l;
                    }
                }
                v[id] = 1;
                r[y[id]]++;
            }
            id = 0;
            l = 0;
            for (j = 0; j < noc; j++) {
                if (r[j] > l) {
                    l = r[j];
                    id = j;
                }
            }
            res[i] = id;
        }
        free(v);
    }
    free(r);
    free(d);
}
