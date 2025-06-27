#include <immintrin.h>
#include <sleef.h>

constexpr int simd_vec_size = 4;

struct FVec {
    FVec() = default;
    FVec(__m256d _v) : v(_v) {};
    FVec(double _v) : v(_mm256_set1_pd(_v)) {};
    explicit FVec(__m256i _v) {
        _v = _mm256_add_epi64(_v, _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000)));
        v = _mm256_sub_pd(_mm256_castsi256_pd(_v), _mm256_set1_pd(0x0018000000000000));
    }
    operator __m256d() { return v; }
    FVec operator+=(FVec _v) { v = _mm256_add_pd(v, _v); return v; }
    __m256d v;
};

struct IVec {
    IVec() = default;
    IVec(__m256i _v) : v(_v) {};
    IVec(int64_t _v) : v(_mm256_set1_epi64x(_v)) {};
    explicit IVec(__m256d _v) {
        _v = _mm256_add_pd(_v, _mm256_set1_pd(0x0018000000000000));
        v = _mm256_sub_epi64(
            _mm256_castpd_si256(_v),
            _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000))
        );
    }
    operator __m256i() { return v; }
    IVec operator+=(IVec _v) { v = _mm256_add_epi64(v, _v); return v; }
    __m256i v;
};

struct BVec {
    BVec() = default;
    BVec(__m256d _v) : v(_v) {};
    BVec(__m256i _v) : v(_mm256_castsi256_pd(_v)) {};
    BVec(bool _v) : v(_mm256_set1_pd(-1 * static_cast<int64_t>(_v))) {};
    operator __m256d() { return v; }
    operator __m256i() { return _mm256_castpd_si256(v); }
    __m256d v;
};

inline __m128i stride_seq(std::size_t stride) {
    return _mm_mul_epi32(_mm_set1_epi32(stride), _mm_set_epi32(3, 2, 1, 0));
}

inline __m128i mem_indices(std::size_t batch_stride, std::size_t index_stride, IVec indices) {
    __m256 vf = _mm256_castsi256_ps(indices);
    __m128 hi = _mm256_extractf128_ps(vf, 1);
    __m128 lo = _mm256_castps256_ps128(vf);
    __m128 indices32 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
    return _mm_add_epi32(
        _mm_mul_epi32(_mm_castps_si128(indices32), _mm_set1_epi32(index_stride)),
        stride_seq(batch_stride)
    );
}

inline FVec vgather(
    double* base_ptr, std::size_t batch_stride, std::size_t index_stride, IVec indices
) {
    return _mm256_i32gather_pd(base_ptr, mem_indices(batch_stride, index_stride, indices), 1);
}

inline IVec vgather(
    int64_t* base_ptr, std::size_t batch_stride, std::size_t index_stride, IVec indices
) {
    return _mm256_i32gather_epi64(
        reinterpret_cast<long long*>(base_ptr),
        mem_indices(batch_stride, index_stride, indices),
        1
    );
}

inline FVec vload(double* base_ptr, std::size_t stride) {
    return _mm256_i32gather_pd(base_ptr, stride_seq(stride), 1);
}

inline IVec vload(int64_t* base_ptr, std::size_t stride) {
    return _mm256_i32gather_epi64(reinterpret_cast<long long*>(base_ptr), stride_seq(stride), 1);
}

inline void vscatter(
    double* base_ptr, std::size_t batch_stride, std::size_t index_stride, IVec indices, FVec values
) {
    _mm256_i32scatter_pd(base_ptr, mem_indices(batch_stride, index_stride, indices), values, 1);
}

inline void vscatter(
    int64_t* base_ptr, std::size_t batch_stride, std::size_t index_stride, IVec indices, IVec values
) {
    _mm256_i32scatter_epi64(base_ptr, mem_indices(batch_stride, index_stride, indices), values, 1);
}

inline void vstore(double* base_ptr, std::size_t stride, FVec values) {
    _mm256_i32scatter_pd(base_ptr, stride_seq(stride), values, 1);
}

inline void vstore(int64_t* base_ptr, std::size_t stride, IVec values) {
    _mm256_i32scatter_epi64(base_ptr, stride_seq(stride), values, 1);
}

inline FVec where(BVec arg1, FVec arg2, FVec arg3) {
    return _mm256_blendv_pd(arg3, arg2, arg1);
}
inline IVec where(BVec arg1, IVec arg2, IVec arg3) {
    return _mm256_blendv_epi8(arg3, arg2, arg1);
}
inline FVec min(FVec arg1, FVec arg2) { return _mm256_min_pd(arg1, arg2); }
inline FVec max(FVec arg1, FVec arg2) { return _mm256_max_pd(arg1, arg2); }
inline std::size_t single_index(IVec arg) { return _mm256_extract_epi64(arg, 0); }

inline BVec operator==(FVec arg1, FVec arg2) { return _mm256_cmp_pd(arg1, arg2, _CMP_EQ_OQ); }
inline BVec operator!=(FVec arg1, FVec arg2) { return _mm256_cmp_pd(arg1, arg2, _CMP_NEQ_UQ); }
inline BVec operator>(FVec arg1, FVec arg2) { return _mm256_cmp_pd(arg1, arg2, _CMP_GT_OQ); }
inline BVec operator<(FVec arg1, FVec arg2) { return _mm256_cmp_pd(arg1, arg2, _CMP_LT_OQ); }
inline BVec operator>=(FVec arg1, FVec arg2) { return _mm256_cmp_pd(arg1, arg2, _CMP_GE_OQ); }
inline BVec operator<=(FVec arg1, FVec arg2) { return _mm256_cmp_pd(arg1, arg2, _CMP_LE_OQ); }

inline BVec operator&(BVec arg1, BVec arg2) { return _mm256_and_si256(arg1, arg2); }
inline BVec operator|(BVec arg1, BVec arg2) { return _mm256_or_si256(arg1, arg2); }
inline BVec operator!(BVec arg1) { return _mm256_xor_si256(arg1, _mm256_cmpeq_epi64(arg1, arg1)); }

inline BVec operator==(IVec arg1, IVec arg2) { return _mm256_cmpeq_epi64(arg1, arg2); }
inline BVec operator!=(IVec arg1, IVec arg2) { return !(arg1 == arg2); }
inline BVec operator>(IVec arg1, IVec arg2) { return _mm256_cmpgt_epi64(arg1, arg2); }
inline BVec operator>=(IVec arg1, IVec arg2) { return (arg1 > arg2) | (arg1 == arg2); }
inline BVec operator<(IVec arg1, IVec arg2) { return !(arg1 >= arg2); }
inline BVec operator<=(IVec arg1, IVec arg2) { return !(arg1 > arg2); }

inline FVec operator-(FVec arg1) { return _mm256_sub_pd(_mm256_set1_pd(0.), arg1); }
inline FVec operator+(FVec arg1, FVec arg2) { return _mm256_add_pd(arg1, arg2); }
inline FVec operator-(FVec arg1, FVec arg2) { return _mm256_sub_pd(arg1, arg2); }
inline FVec operator*(FVec arg1, FVec arg2) { return _mm256_mul_pd(arg1, arg2); }
inline FVec operator/(FVec arg1, FVec arg2) { return _mm256_div_pd(arg1, arg2); }
inline IVec operator-(IVec arg1) { return _mm256_sub_epi64(_mm256_set1_epi64x(0), arg1); }
inline IVec operator+(IVec arg1, IVec arg2) { return _mm256_add_epi64(arg1, arg2); }
inline IVec operator-(IVec arg1, IVec arg2) { return _mm256_sub_epi64(arg1, arg2); }

inline FVec sqrt(FVec arg1) { return Sleef_sqrtd4_u05avx2(arg1); }
inline FVec sin(FVec arg1) { return Sleef_sind4_u10avx2(arg1); }
inline FVec cos(FVec arg1) { return Sleef_cosd4_u10avx2(arg1); }
inline FVec sinh(FVec arg1) { return Sleef_sinhd4_u10avx2(arg1); }
inline FVec cosh(FVec arg1) { return Sleef_coshd4_u10avx2(arg1); }
inline FVec atan2(FVec arg1, FVec arg2) { return Sleef_atan2d4_u10avx2(arg1, arg2); }
inline FVec pow(FVec arg1, FVec arg2) { return Sleef_powd4_u10avx2(arg1, arg2); }
inline FVec fabs(FVec arg1) { return Sleef_fabsd4_avx2(arg1); }
inline FVec log(FVec arg1) { return Sleef_logd4_u10avx2(arg1); }
inline FVec tan(FVec arg1) { return Sleef_tand4_u10avx2(arg1); }
inline FVec atan(FVec arg1) { return Sleef_atand4_u10avx2(arg1); }
inline FVec exp(FVec arg1) { return Sleef_expd4_u10avx2(arg1); }
inline FVec log1p(FVec arg1) { return Sleef_log1pd4_u10avx2(arg1); }

inline BVec isnan(FVec arg) { return arg != arg; }
