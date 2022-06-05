#include <stdio.h>
#include <immintrin.h>

float sdot(size_t n, const float x[n], const float y[n]) {
    __m128 acc = _mm_setzero_ps();
    const size_t n4 = n-n%4;
    for (int i = 0; i < n4; i += 4)
    {
        __m128 xm = _mm_loadu_ps(&x[i]);
        __m128 ym = _mm_loadu_ps(&y[i]);
        __m128 zm = _mm_mul_ps(xm, ym);
        acc = _mm_add_ps(acc,zm);
    }
    float res[4];
    _mm_storeu_ps(&res[0], acc);
    for (size_t i = n4; i < n; ++i) {
        res[i-n4] += x[i]*y[i];
    }
    return res[0]+res[1]+res[2]+res[3];
}

int main() {
    float a[] = {1,2,3,4};
    float b[] = {2,2,2,2};
    printf("%f", sdot(4,a,b));
    return 0;
}
