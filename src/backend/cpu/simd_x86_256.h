#include <immintrin.h>
#include <sleef.h>

constexpr int simd_vec_size = 4;

struct FVec {
    FVec() = default;
    FVec(__m256d _v) : v(_v) {};
    FVec(double _v) : v(vdupq_n_f64(_v)) {};
    operator __m256d() { return v; }
    __m256d v;
};

struct IVec {
    IVec() = default;
    IVec(__mm256i _v) : v(_v) {};
    IVec(long long _v) : v(vdupq_n_s64(_v)) {};
    operator __mm256i() { return v; }
    __mm256i v;
};

struct BVec {
    BVec() = default;
    BVec(uint64x2_t _v) : v(_v) {};
    BVec(bool _v) : v(vceqzq_u64(vdupq_n_u64(_v))) {};
    operator uint64x2_t() { return v; }
    uint64x2_t v;
};

FVec vload(float64_t* ptr) {
    return _mm256_loadu_pd(ptr);
}

IVec vload(int64_t* ptr) {
    return _mm256_loadu_si256(ptr);
}

BVec vload(bool* ptr) {
    uint64_t buffer[2]{ptr[0], ptr[1]};
    return vld1q_u64(&buffer[0]);
}

void vstore(float64_t* ptr, FVec values) {
    vst1q_f64(ptr, values);
}

void vstore(int64_t* ptr, IVec values) {
    vst1q_s64(ptr, values);
}

void vstore(bool* ptr, BVec values) {
    uint64_t buffer[2];
    vst1q_u64(&buffer[0], values);
    ptr[0] = buffer[0] != 0;
    ptr[1] = buffer[1] != 0;
}

FVec where(BVec arg1, FVec arg2, FVec arg3) {
    return vbslq_f64(arg1, arg2, arg3);
}
BVec operator==(FVec arg1, FVec arg2) { return _mm256_cmp_pd(arg1, arg2, _CMP_EQ_OQ); }
BVec operator!=(FVec arg1, FVec arg2) { return _mm256_cmp_pd(arg1, arg2, _CMP_NEQ_UQ); }
BVec operator>(FVec arg1, FVec arg2) { return _mm256_cmp_pd(arg1, arg2, _CMP_GT_OQ); }
BVec operator<(FVec arg1, FVec arg2) { return _mm256_cmp_pd(arg1, arg2, _CMP_LT_OQ); }
BVec operator>=(FVec arg1, FVec arg2) { return _mm256_cmp_pd(arg1, arg2, _CMP_GE_OQ); }
BVec operator<=(FVec arg1, FVec arg2) { return _mm256_cmp_pd(arg1, arg2, _CMP_LE_OQ); }
BVec operator&(BVec arg1, BVec arg2) { return vandq_u64(arg1, arg2); }
BVec operator|(BVec arg1, BVec arg2) { return vorrq_u64(arg1, arg2); }
BVec operator!(BVec arg1) { return vceqzq_u64(arg1); }
FVec operator-(FVec arg1) { return vnegq_f64(arg1); }
FVec operator+(FVec arg1, FVec arg2) { return _mm256_add_pd(arg1, arg2); }
FVec operator-(FVec arg1, FVec arg2) { return _mm256_sub_pd(arg1, arg2); }
FVec operator*(FVec arg1, FVec arg2) { return _mm256_mul_pd(arg1, arg2); }
FVec operator/(FVec arg1, FVec arg2) { return _mm256_div_pd(arg1, arg2); }
IVec operator-(IVec arg1) { return vnegq_s64(arg1); }
IVec operator+(IVec arg1, IVec arg2) { return vaddq_s64(arg1, arg2); }
IVec operator-(IVec arg1, IVec arg2) { return vsubq_s64(arg1, arg2); }
//IVec operator*(IVec arg1, IVec arg2) { return vmulq_s64(arg1, arg2); }
FVec sqrt(FVec arg1) { return Sleef_sqrtd4_u05(arg1); }
FVec sin(FVec arg1) { return Sleef_sind4_u10(arg1); }
FVec cos(FVec arg1) { return Sleef_cosd4_u10(arg1); }
FVec sinh(FVec arg1) { return Sleef_sinhd4_u10(arg1); }
FVec cosh(FVec arg1) { return Sleef_coshd4_u10(arg1); }
FVec atan2(FVec arg1, FVec arg2) { return Sleef_atan2d4_u10(arg1, arg2); }
FVec pow(FVec arg1, FVec arg2) { return Sleef_powd4_u10(arg1, arg2); }
FVec fabs(FVec arg1) { return Sleef_fabsd4(arg1); }
FVec log(FVec arg1) { return Sleef_logd4_u10(arg1); }
FVec tan(FVec arg1) { return Sleef_tand4_u10(arg1); }
FVec atan(FVec arg1) { return Sleef_atand4_u10(arg1); }
FVec exp(FVec arg1) { return Sleef_expd4_u10(arg1); }
