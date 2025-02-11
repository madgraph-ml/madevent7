#include <arm_neon.h>
#include <sleef.h>

constexpr int simd_vec_size = 2;

struct FVec {
    FVec() = default;
    FVec(float64x2_t _v) : v(_v) {};
    FVec(double _v) : v(vdupq_n_f64(_v)) {};
    operator float64x2_t() { return v; }
    float64x2_t v;
};

struct IVec {
    IVec() = default;
    IVec(int64x2_t _v) : v(_v) {};
    IVec(long long _v) : v(vdupq_n_s64(_v)) {};
    operator int64x2_t() { return v; }
    int64x2_t v;
};

struct BVec {
    BVec() = default;
    BVec(uint64x2_t _v) : v(_v) {};
    BVec(bool _v) : v(vceqzq_u64(vdupq_n_u64(_v))) {};
    operator uint64x2_t() { return v; }
    uint64x2_t v;
};

FVec vload(float64_t* ptr) {
    return vld1q_f64(ptr);
}

IVec vload(int64_t* ptr) {
    return vld1q_s64(ptr);
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
std::size_t single_index(IVec arg) {
    return vgetq_lane_s64(arg, 0);
}
BVec operator==(FVec arg1, FVec arg2) { return vceqq_f64(arg1, arg2); }
BVec operator!=(FVec arg1, FVec arg2) { return vceqzq_u64(vceqq_f64(arg1, arg2)); }
BVec operator>(FVec arg1, FVec arg2) { return vcgtq_f64(arg1, arg2); }
BVec operator<(FVec arg1, FVec arg2) { return vcltq_f64(arg1, arg2); }
BVec operator>=(FVec arg1, FVec arg2) { return vcgeq_f64(arg1, arg2); }
BVec operator<=(FVec arg1, FVec arg2) { return vcleq_f64(arg1, arg2); }
BVec operator&(BVec arg1, BVec arg2) { return vandq_u64(arg1, arg2); }
BVec operator|(BVec arg1, BVec arg2) { return vorrq_u64(arg1, arg2); }
BVec operator!(BVec arg1) { return vceqzq_u64(arg1); }
FVec operator-(FVec arg1) { return vnegq_f64(arg1); }
FVec operator+(FVec arg1, FVec arg2) { return vaddq_f64(arg1, arg2); }
FVec operator-(FVec arg1, FVec arg2) { return vsubq_f64(arg1, arg2); }
FVec operator*(FVec arg1, FVec arg2) { return vmulq_f64(arg1, arg2); }
FVec operator/(FVec arg1, FVec arg2) { return vdivq_f64(arg1, arg2); }
IVec operator-(IVec arg1) { return vnegq_s64(arg1); }
IVec operator+(IVec arg1, IVec arg2) { return vaddq_s64(arg1, arg2); }
IVec operator-(IVec arg1, IVec arg2) { return vsubq_s64(arg1, arg2); }
//IVec operator*(IVec arg1, IVec arg2) { return vmulq_s64(arg1, arg2); }
FVec sqrt(FVec arg1) { return Sleef_sqrtd2_u05(arg1); }
FVec sin(FVec arg1) { return Sleef_sind2_u10(arg1); }
FVec cos(FVec arg1) { return Sleef_cosd2_u10(arg1); }
FVec sinh(FVec arg1) { return Sleef_sinhd2_u10(arg1); }
FVec cosh(FVec arg1) { return Sleef_coshd2_u10(arg1); }
FVec atan2(FVec arg1, FVec arg2) { return Sleef_atan2d2_u10(arg1, arg2); }
FVec pow(FVec arg1, FVec arg2) { return Sleef_powd2_u10(arg1, arg2); }
FVec fabs(FVec arg1) { return Sleef_fabsd2(arg1); }
FVec log(FVec arg1) { return Sleef_logd2_u10(arg1); }
FVec tan(FVec arg1) { return Sleef_tand2_u10(arg1); }
FVec atan(FVec arg1) { return Sleef_atand2_u10(arg1); }
FVec exp(FVec arg1) { return Sleef_expd2_u10(arg1); }

bool isnan(FVec arg) { return false; }
