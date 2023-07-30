#include <stdio.h>

double pi_squared_over_6(unsigned int N) {
    double sum = 0.0;
    for (int i = 1; i < N; i++) {
        sum += 1.0 / ((double)i*i);
    }
    return sum;
}

int main() {
    printf("%f\n", pi_squared_over_6(100000000));
}
