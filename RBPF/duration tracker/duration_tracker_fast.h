/* duration_tracker_fast.h */
#ifndef DURATION_TRACKER_FAST_H
#define DURATION_TRACKER_FAST_H

#include <stdint.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════
   SIMD Feature Detection
   ═══════════════════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    #define DT_USE_AVX512 1
    #define DT_USE_AVX2   1
    #include <immintrin.h>
#elif defined(__AVX2__)
    #define DT_USE_AVX512 0
    #define DT_USE_AVX2   1
    #include <immintrin.h>
#else
    #define DT_USE_AVX512 0
    #define DT_USE_AVX2   0
#endif

/* Manual override: compile with -DDT_FORCE_SCALAR=1 to disable SIMD */
#ifdef DT_FORCE_SCALAR
    #undef DT_USE_AVX512
    #undef DT_USE_AVX2
    #define DT_USE_AVX512 0
    #define DT_USE_AVX2   0
#endif

#define DT_MAX_REGIMES   4
#define DT_MAX_DURATION  2000
#define DT_CACHE_LINE    64

/* LUT indices */
#define DT_HAZARD   0
#define DT_LOG_HAZ  1
#define DT_SURVIVAL 2
#define DT_EXPECTED 3

/* ═══════════════════════════════════════════════════════════════════════════
   Data Structures
   ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Interleaved: [hazard, log_hazard, survival, expected] per (regime, duration) */
    float data[DT_MAX_REGIMES][DT_MAX_DURATION][4] __attribute__((aligned(64)));
} DurationLUT;

typedef struct {
    int32_t  current_regime;
    int32_t  ticks_in_regime;
    int32_t  prev_regime;
    int32_t  _pad;
    uint64_t total_transitions;
    const DurationLUT *lut;
} DurationTracker;

typedef struct {
    float hazard;
    float log_hazard;
    float survival;
    float expected_remaining;
} DurationFeatures4;

/* ═══════════════════════════════════════════════════════════════════════════
   Fast Math - Scalar
   ═══════════════════════════════════════════════════════════════════════════ */

static inline float fast_log_scalar(float x) {
    union { float f; uint32_t i; } vx = { x };
    float e = (float)((int32_t)(vx.i >> 23) - 127);
    vx.i = (vx.i & 0x007FFFFF) | 0x3F800000;
    float m = vx.f;
    float p = -1.49278297f;
    p = p * m + 5.52591772f;
    p = p * m - 7.76753640f;
    p = p * m + 4.73063259f;
    return p + e * 0.6931471805599453f;
}

static inline float fast_fmaxf(float a, float b) {
    return (a > b) ? a : b;
}

static inline float fast_fminf(float a, float b) {
    return (a < b) ? a : b;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Fast Math - AVX2
   ═══════════════════════════════════════════════════════════════════════════ */

#if DT_USE_AVX2

static inline __m256 fast_log_avx2(__m256 x) {
    /* 
     * Vectorized IEEE 754 log approximation
     * Process 8 floats in parallel
     */
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 c0 = _mm256_set1_ps(-1.49278297f);
    const __m256 c1 = _mm256_set1_ps(5.52591772f);
    const __m256 c2 = _mm256_set1_ps(-7.76753640f);
    const __m256 c3 = _mm256_set1_ps(4.73063259f);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256i exp_mask = _mm256_set1_epi32(0x7F800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007FFFFF);
    const __m256i bias = _mm256_set1_epi32(127);
    const __m256i one_bits = _mm256_set1_epi32(0x3F800000);
    
    __m256i xi = _mm256_castps_si256(x);
    
    /* Extract exponent: e = ((xi >> 23) & 0xFF) - 127 */
    __m256i exp_bits = _mm256_srli_epi32(xi, 23);
    __m256i e_int = _mm256_sub_epi32(exp_bits, bias);
    __m256 e = _mm256_cvtepi32_ps(e_int);
    
    /* Normalize mantissa to [1, 2) */
    __m256i mant = _mm256_and_si256(xi, mant_mask);
    __m256i m_bits = _mm256_or_si256(mant, one_bits);
    __m256 m = _mm256_castsi256_ps(m_bits);
    
    /* Polynomial: p = c0*m^3 + c1*m^2 + c2*m + c3 */
    __m256 p = c0;
    p = _mm256_fmadd_ps(p, m, c1);
    p = _mm256_fmadd_ps(p, m, c2);
    p = _mm256_fmadd_ps(p, m, c3);
    
    /* Result: p + e * ln(2) */
    return _mm256_fmadd_ps(e, ln2, p);
}

static inline __m256 fast_fmax_avx2(__m256 a, __m256 b) {
    return _mm256_max_ps(a, b);
}

static inline __m256 fast_fmin_avx2(__m256 a, __m256 b) {
    return _mm256_min_ps(a, b);
}

#endif /* DT_USE_AVX2 */

/* ═══════════════════════════════════════════════════════════════════════════
   Fast Math - AVX-512
   ═══════════════════════════════════════════════════════════════════════════ */

#if DT_USE_AVX512

static inline __m512 fast_log_avx512(__m512 x) {
    /* 
     * Vectorized IEEE 754 log approximation
     * Process 16 floats in parallel
     */
    const __m512 c0 = _mm512_set1_ps(-1.49278297f);
    const __m512 c1 = _mm512_set1_ps(5.52591772f);
    const __m512 c2 = _mm512_set1_ps(-7.76753640f);
    const __m512 c3 = _mm512_set1_ps(4.73063259f);
    const __m512 ln2 = _mm512_set1_ps(0.6931471805599453f);
    const __m512i bias = _mm512_set1_epi32(127);
    const __m512i mant_mask = _mm512_set1_epi32(0x007FFFFF);
    const __m512i one_bits = _mm512_set1_epi32(0x3F800000);
    
    __m512i xi = _mm512_castps_si512(x);
    
    /* Extract exponent */
    __m512i exp_bits = _mm512_srli_epi32(xi, 23);
    __m512i e_int = _mm512_sub_epi32(exp_bits, bias);
    __m512 e = _mm512_cvtepi32_ps(e_int);
    
    /* Normalize mantissa */
    __m512i mant = _mm512_and_si512(xi, mant_mask);
    __m512i m_bits = _mm512_or_si512(mant, one_bits);
    __m512 m = _mm512_castsi512_ps(m_bits);
    
    /* Polynomial with FMA */
    __m512 p = c0;
    p = _mm512_fmadd_ps(p, m, c1);
    p = _mm512_fmadd_ps(p, m, c2);
    p = _mm512_fmadd_ps(p, m, c3);
    
    return _mm512_fmadd_ps(e, ln2, p);
}

static inline __m512 fast_fmax_avx512(__m512 a, __m512 b) {
    return _mm512_max_ps(a, b);
}

static inline __m512 fast_fmin_avx512(__m512 a, __m512 b) {
    return _mm512_min_ps(a, b);
}

#endif /* DT_USE_AVX512 */

/* ═══════════════════════════════════════════════════════════════════════════
   Dispatch Macros - Auto-select best implementation
   ═══════════════════════════════════════════════════════════════════════════ */

#if DT_USE_AVX512
    #define fast_log_8      fast_log_avx2
    #define fast_log_16     fast_log_avx512
    #define DT_SIMD_WIDTH   16
    #define DT_VEC_PS       __m512
    #define DT_VEC_EPI32    __m512i
#elif DT_USE_AVX2
    #define fast_log_8      fast_log_avx2
    #define DT_SIMD_WIDTH   8
    #define DT_VEC_PS       __m256
    #define DT_VEC_EPI32    __m256i
#else
    #define DT_SIMD_WIDTH   1
#endif

/* ═══════════════════════════════════════════════════════════════════════════
   Update - Scalar (branchless)
   ═══════════════════════════════════════════════════════════════════════════ */

static inline void dt_update_scalar(DurationTracker *dt, int regime) {
    int same = (regime == dt->current_regime);
    int new_ticks = (dt->ticks_in_regime + 1) * same + (1 - same);
    int clamped = new_ticks - (new_ticks >= DT_MAX_DURATION) * (new_ticks - DT_MAX_DURATION + 1);
    
    dt->prev_regime = dt->current_regime;
    dt->current_regime = regime;
    dt->ticks_in_regime = clamped;
    dt->total_transitions += (uint64_t)(1 - same);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Update - AVX2 (8 particles in parallel)
   ═══════════════════════════════════════════════════════════════════════════ */

#if DT_USE_AVX2

static inline void dt_update_8_avx2(int32_t *regimes_cur,      /* [8] in/out */
                                     int32_t *regimes_prev,     /* [8] out */
                                     int32_t *ticks,            /* [8] in/out */
                                     const int32_t *new_regimes /* [8] in */) {
    __m256i cur = _mm256_loadu_si256((const __m256i*)regimes_cur);
    __m256i new_r = _mm256_loadu_si256((const __m256i*)new_regimes);
    __m256i t = _mm256_loadu_si256((const __m256i*)ticks);
    
    /* same = (cur == new_r) ? 0xFFFFFFFF : 0 */
    __m256i same = _mm256_cmpeq_epi32(cur, new_r);
    
    /* new_ticks = same ? (ticks + 1) : 1 */
    __m256i one = _mm256_set1_epi32(1);
    __m256i t_inc = _mm256_add_epi32(t, one);
    __m256i new_t = _mm256_blendv_epi8(one, t_inc, same);
    
    /* Clamp to MAX_DURATION - 1 */
    __m256i max_dur = _mm256_set1_epi32(DT_MAX_DURATION - 1);
    new_t = _mm256_min_epi32(new_t, max_dur);
    
    /* Store results */
    _mm256_storeu_si256((__m256i*)regimes_prev, cur);
    _mm256_storeu_si256((__m256i*)regimes_cur, new_r);
    _mm256_storeu_si256((__m256i*)ticks, new_t);
}

#endif /* DT_USE_AVX2 */

/* ═══════════════════════════════════════════════════════════════════════════
   Update - AVX-512 (16 particles in parallel)
   ═══════════════════════════════════════════════════════════════════════════ */

#if DT_USE_AVX512

static inline void dt_update_16_avx512(int32_t *regimes_cur,
                                        int32_t *regimes_prev,
                                        int32_t *ticks,
                                        const int32_t *new_regimes) {
    __m512i cur = _mm512_loadu_si512((const __m512i*)regimes_cur);
    __m512i new_r = _mm512_loadu_si512((const __m512i*)new_regimes);
    __m512i t = _mm512_loadu_si512((const __m512i*)ticks);
    
    /* same mask: cur == new_r */
    __mmask16 same = _mm512_cmpeq_epi32_mask(cur, new_r);
    
    /* new_ticks = same ? (ticks + 1) : 1 */
    __m512i one = _mm512_set1_epi32(1);
    __m512i t_inc = _mm512_add_epi32(t, one);
    __m512i new_t = _mm512_mask_blend_epi32(same, one, t_inc);
    
    /* Clamp */
    __m512i max_dur = _mm512_set1_epi32(DT_MAX_DURATION - 1);
    new_t = _mm512_min_epi32(new_t, max_dur);
    
    _mm512_storeu_si512((__m512i*)regimes_prev, cur);
    _mm512_storeu_si512((__m512i*)regimes_cur, new_r);
    _mm512_storeu_si512((__m512i*)ticks, new_t);
}

#endif /* DT_USE_AVX512 */

/* ═══════════════════════════════════════════════════════════════════════════
   Feature Lookup - Scalar
   ═══════════════════════════════════════════════════════════════════════════ */

static inline void dt_get_features4_scalar(const DurationTracker *dt, 
                                            DurationFeatures4 *f) {
    const float *src = dt->lut->data[dt->current_regime][dt->ticks_in_regime];
    f->hazard            = src[DT_HAZARD];
    f->log_hazard        = src[DT_LOG_HAZ];
    f->survival          = src[DT_SURVIVAL];
    f->expected_remaining = src[DT_EXPECTED];
}

static inline float dt_hazard_scalar(const DurationTracker *dt) {
    return dt->lut->data[dt->current_regime][dt->ticks_in_regime][DT_HAZARD];
}

static inline float dt_log_hazard_scalar(const DurationTracker *dt) {
    return dt->lut->data[dt->current_regime][dt->ticks_in_regime][DT_LOG_HAZ];
}

static inline float dt_survival_scalar(const DurationTracker *dt) {
    return dt->lut->data[dt->current_regime][dt->ticks_in_regime][DT_SURVIVAL];
}

static inline float dt_expected_scalar(const DurationTracker *dt) {
    return dt->lut->data[dt->current_regime][dt->ticks_in_regime][DT_EXPECTED];
}

/* ═══════════════════════════════════════════════════════════════════════════
   Feature Lookup - AVX2 Gather (8 particles)
   ═══════════════════════════════════════════════════════════════════════════ */

#if DT_USE_AVX2

static inline void dt_get_hazard_8_avx2(const DurationLUT *lut,
                                         const int32_t *regimes,
                                         const int32_t *durations,
                                         float *hazards_out) {
    /* 
     * Compute gather indices: r * (DT_MAX_DURATION * 4) + d * 4 + DT_HAZARD
     * Each (r,d) entry is 4 floats, hazard is at offset 0
     */
    __m256i r = _mm256_loadu_si256((const __m256i*)regimes);
    __m256i d = _mm256_loadu_si256((const __m256i*)durations);
    
    __m256i stride_r = _mm256_set1_epi32(DT_MAX_DURATION * 4);
    __m256i stride_d = _mm256_set1_epi32(4);
    
    __m256i idx = _mm256_add_epi32(
        _mm256_mullo_epi32(r, stride_r),
        _mm256_mullo_epi32(d, stride_d)
    );
    /* Add DT_HAZARD offset (0, so no-op) */
    
    __m256 result = _mm256_i32gather_ps((const float*)lut->data, idx, 4);
    _mm256_storeu_ps(hazards_out, result);
}

static inline void dt_get_log_hazard_8_avx2(const DurationLUT *lut,
                                             const int32_t *regimes,
                                             const int32_t *durations,
                                             float *log_hazards_out) {
    __m256i r = _mm256_loadu_si256((const __m256i*)regimes);
    __m256i d = _mm256_loadu_si256((const __m256i*)durations);
    
    __m256i stride_r = _mm256_set1_epi32(DT_MAX_DURATION * 4);
    __m256i stride_d = _mm256_set1_epi32(4);
    __m256i offset = _mm256_set1_epi32(DT_LOG_HAZ);
    
    __m256i idx = _mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(r, stride_r),
            _mm256_mullo_epi32(d, stride_d)
        ),
        offset
    );
    
    __m256 result = _mm256_i32gather_ps((const float*)lut->data, idx, 4);
    _mm256_storeu_ps(log_hazards_out, result);
}

static inline void dt_get_survival_8_avx2(const DurationLUT *lut,
                                           const int32_t *regimes,
                                           const int32_t *durations,
                                           float *survival_out) {
    __m256i r = _mm256_loadu_si256((const __m256i*)regimes);
    __m256i d = _mm256_loadu_si256((const __m256i*)durations);
    
    __m256i stride_r = _mm256_set1_epi32(DT_MAX_DURATION * 4);
    __m256i stride_d = _mm256_set1_epi32(4);
    __m256i offset = _mm256_set1_epi32(DT_SURVIVAL);
    
    __m256i idx = _mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(r, stride_r),
            _mm256_mullo_epi32(d, stride_d)
        ),
        offset
    );
    
    __m256 result = _mm256_i32gather_ps((const float*)lut->data, idx, 4);
    _mm256_storeu_ps(survival_out, result);
}

static inline void dt_get_expected_8_avx2(const DurationLUT *lut,
                                           const int32_t *regimes,
                                           const int32_t *durations,
                                           float *expected_out) {
    __m256i r = _mm256_loadu_si256((const __m256i*)regimes);
    __m256i d = _mm256_loadu_si256((const __m256i*)durations);
    
    __m256i stride_r = _mm256_set1_epi32(DT_MAX_DURATION * 4);
    __m256i stride_d = _mm256_set1_epi32(4);
    __m256i offset = _mm256_set1_epi32(DT_EXPECTED);
    
    __m256i idx = _mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(r, stride_r),
            _mm256_mullo_epi32(d, stride_d)
        ),
        offset
    );
    
    __m256 result = _mm256_i32gather_ps((const float*)lut->data, idx, 4);
    _mm256_storeu_ps(expected_out, result);
}

/* All 4 features at once for 8 particles */
static inline void dt_get_all_features_8_avx2(const DurationLUT *lut,
                                               const int32_t *regimes,
                                               const int32_t *durations,
                                               float *hazard_out,
                                               float *log_hazard_out,
                                               float *survival_out,
                                               float *expected_out) {
    __m256i r = _mm256_loadu_si256((const __m256i*)regimes);
    __m256i d = _mm256_loadu_si256((const __m256i*)durations);
    
    __m256i stride_r = _mm256_set1_epi32(DT_MAX_DURATION * 4);
    __m256i stride_d = _mm256_set1_epi32(4);
    
    __m256i base_idx = _mm256_add_epi32(
        _mm256_mullo_epi32(r, stride_r),
        _mm256_mullo_epi32(d, stride_d)
    );
    
    /* 4 gathers with different offsets */
    __m256i off0 = _mm256_set1_epi32(DT_HAZARD);
    __m256i off1 = _mm256_set1_epi32(DT_LOG_HAZ);
    __m256i off2 = _mm256_set1_epi32(DT_SURVIVAL);
    __m256i off3 = _mm256_set1_epi32(DT_EXPECTED);
    
    const float *base = (const float*)lut->data;
    
    _mm256_storeu_ps(hazard_out,     _mm256_i32gather_ps(base, _mm256_add_epi32(base_idx, off0), 4));
    _mm256_storeu_ps(log_hazard_out, _mm256_i32gather_ps(base, _mm256_add_epi32(base_idx, off1), 4));
    _mm256_storeu_ps(survival_out,   _mm256_i32gather_ps(base, _mm256_add_epi32(base_idx, off2), 4));
    _mm256_storeu_ps(expected_out,   _mm256_i32gather_ps(base, _mm256_add_epi32(base_idx, off3), 4));
}

#endif /* DT_USE_AVX2 */

/* ═══════════════════════════════════════════════════════════════════════════
   Feature Lookup - AVX-512 Gather (16 particles)
   ═══════════════════════════════════════════════════════════════════════════ */

#if DT_USE_AVX512

static inline void dt_get_hazard_16_avx512(const DurationLUT *lut,
                                            const int32_t *regimes,
                                            const int32_t *durations,
                                            float *hazards_out) {
    __m512i r = _mm512_loadu_si512((const __m512i*)regimes);
    __m512i d = _mm512_loadu_si512((const __m512i*)durations);
    
    __m512i stride_r = _mm512_set1_epi32(DT_MAX_DURATION * 4);
    __m512i stride_d = _mm512_set1_epi32(4);
    
    __m512i idx = _mm512_add_epi32(
        _mm512_mullo_epi32(r, stride_r),
        _mm512_mullo_epi32(d, stride_d)
    );
    
    __m512 result = _mm512_i32gather_ps(idx, (const float*)lut->data, 4);
    _mm512_storeu_ps(hazards_out, result);
}

static inline void dt_get_log_hazard_16_avx512(const DurationLUT *lut,
                                                const int32_t *regimes,
                                                const int32_t *durations,
                                                float *log_hazards_out) {
    __m512i r = _mm512_loadu_si512((const __m512i*)regimes);
    __m512i d = _mm512_loadu_si512((const __m512i*)durations);
    
    __m512i stride_r = _mm512_set1_epi32(DT_MAX_DURATION * 4);
    __m512i stride_d = _mm512_set1_epi32(4);
    __m512i offset = _mm512_set1_epi32(DT_LOG_HAZ);
    
    __m512i idx = _mm512_add_epi32(
        _mm512_add_epi32(
            _mm512_mullo_epi32(r, stride_r),
            _mm512_mullo_epi32(d, stride_d)
        ),
        offset
    );
    
    __m512 result = _mm512_i32gather_ps(idx, (const float*)lut->data, 4);
    _mm512_storeu_ps(log_hazards_out, result);
}

static inline void dt_get_all_features_16_avx512(const DurationLUT *lut,
                                                  const int32_t *regimes,
                                                  const int32_t *durations,
                                                  float *hazard_out,
                                                  float *log_hazard_out,
                                                  float *survival_out,
                                                  float *expected_out) {
    __m512i r = _mm512_loadu_si512((const __m512i*)regimes);
    __m512i d = _mm512_loadu_si512((const __m512i*)durations);
    
    __m512i stride_r = _mm512_set1_epi32(DT_MAX_DURATION * 4);
    __m512i stride_d = _mm512_set1_epi32(4);
    
    __m512i base_idx = _mm512_add_epi32(
        _mm512_mullo_epi32(r, stride_r),
        _mm512_mullo_epi32(d, stride_d)
    );
    
    __m512i off0 = _mm512_set1_epi32(DT_HAZARD);
    __m512i off1 = _mm512_set1_epi32(DT_LOG_HAZ);
    __m512i off2 = _mm512_set1_epi32(DT_SURVIVAL);
    __m512i off3 = _mm512_set1_epi32(DT_EXPECTED);
    
    const float *base = (const float*)lut->data;
    
    _mm512_storeu_ps(hazard_out,     _mm512_i32gather_ps(_mm512_add_epi32(base_idx, off0), base, 4));
    _mm512_storeu_ps(log_hazard_out, _mm512_i32gather_ps(_mm512_add_epi32(base_idx, off1), base, 4));
    _mm512_storeu_ps(survival_out,   _mm512_i32gather_ps(_mm512_add_epi32(base_idx, off2), base, 4));
    _mm512_storeu_ps(expected_out,   _mm512_i32gather_ps(_mm512_add_epi32(base_idx, off3), base, 4));
}

#endif /* DT_USE_AVX512 */

/* ═══════════════════════════════════════════════════════════════════════════
   Kelly Multiplier - Scalar (branchless)
   ═══════════════════════════════════════════════════════════════════════════ */

static inline float dt_kelly_mult_scalar(const DurationTracker *dt,
                                          float inv_expected_duration) {
    float age = (float)dt->ticks_in_regime * inv_expected_duration;
    float mult = 1.5f - 0.5f * age;
    mult = fast_fmaxf(0.3f, fast_fminf(1.0f, mult));
    float hazard = dt->lut->data[dt->current_regime][dt->ticks_in_regime][DT_HAZARD];
    return mult * (1.0f - hazard);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Kelly Multiplier - AVX2 (8 particles)
   ═══════════════════════════════════════════════════════════════════════════ */

#if DT_USE_AVX2

static inline void dt_kelly_mult_8_avx2(const DurationLUT *lut,
                                         const int32_t *regimes,
                                         const int32_t *durations,
                                         const float *inv_expected,  /* [4] per regime */
                                         float *kelly_out) {
    /* Load durations as float */
    __m256i d_int = _mm256_loadu_si256((const __m256i*)durations);
    __m256 d = _mm256_cvtepi32_ps(d_int);
    
    /* Gather inv_expected per particle's regime */
    __m256i r = _mm256_loadu_si256((const __m256i*)regimes);
    __m256 inv_exp = _mm256_i32gather_ps(inv_expected, r, 4);
    
    /* age = d * inv_expected */
    __m256 age = _mm256_mul_ps(d, inv_exp);
    
    /* mult = clamp(1.5 - 0.5 * age, 0.3, 1.0) */
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one_five = _mm256_set1_ps(1.5f);
    __m256 floor_val = _mm256_set1_ps(0.3f);
    __m256 ceil_val = _mm256_set1_ps(1.0f);
    
    __m256 mult = _mm256_fnmadd_ps(half, age, one_five);  /* 1.5 - 0.5*age */
    mult = _mm256_max_ps(floor_val, _mm256_min_ps(ceil_val, mult));
    
    /* Get hazards via gather */
    __m256i stride_r = _mm256_set1_epi32(DT_MAX_DURATION * 4);
    __m256i stride_d = _mm256_set1_epi32(4);
    __m256i idx = _mm256_add_epi32(
        _mm256_mullo_epi32(r, stride_r),
        _mm256_mullo_epi32(d_int, stride_d)
    );
    __m256 hazard = _mm256_i32gather_ps((const float*)lut->data, idx, 4);
    
    /* kelly = mult * (1 - hazard) */
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 stability = _mm256_sub_ps(one, hazard);
    __m256 kelly = _mm256_mul_ps(mult, stability);
    
    _mm256_storeu_ps(kelly_out, kelly);
}

#endif /* DT_USE_AVX2 */

/* ═══════════════════════════════════════════════════════════════════════════
   Kelly Multiplier - AVX-512 (16 particles)
   ═══════════════════════════════════════════════════════════════════════════ */

#if DT_USE_AVX512

static inline void dt_kelly_mult_16_avx512(const DurationLUT *lut,
                                            const int32_t *regimes,
                                            const int32_t *durations,
                                            const float *inv_expected,
                                            float *kelly_out) {
    __m512i d_int = _mm512_loadu_si512((const __m512i*)durations);
    __m512 d = _mm512_cvtepi32_ps(d_int);
    
    __m512i r = _mm512_loadu_si512((const __m512i*)regimes);
    __m512 inv_exp = _mm512_i32gather_ps(r, inv_expected, 4);
    
    __m512 age = _mm512_mul_ps(d, inv_exp);
    
    __m512 half = _mm512_set1_ps(0.5f);
    __m512 one_five = _mm512_set1_ps(1.5f);
    __m512 floor_val = _mm512_set1_ps(0.3f);
    __m512 ceil_val = _mm512_set1_ps(1.0f);
    
    __m512 mult = _mm512_fnmadd_ps(half, age, one_five);
    mult = _mm512_max_ps(floor_val, _mm512_min_ps(ceil_val, mult));
    
    __m512i stride_r = _mm512_set1_epi32(DT_MAX_DURATION * 4);
    __m512i stride_d = _mm512_set1_epi32(4);
    __m512i idx = _mm512_add_epi32(
        _mm512_mullo_epi32(r, stride_r),
        _mm512_mullo_epi32(d_int, stride_d)
    );
    __m512 hazard = _mm512_i32gather_ps(idx, (const float*)lut->data, 4);
    
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 stability = _mm512_sub_ps(one, hazard);
    __m512 kelly = _mm512_mul_ps(mult, stability);
    
    _mm512_storeu_ps(kelly_out, kelly);
}

#endif /* DT_USE_AVX512 */

/* ═══════════════════════════════════════════════════════════════════════════
   Dispatch Macros - Single API, best implementation auto-selected
   ═══════════════════════════════════════════════════════════════════════════ */

/* Single tracker (always scalar - no benefit from SIMD for single lookup) */
#define dt_update           dt_update_scalar
#define dt_get_features4    dt_get_features4_scalar
#define dt_hazard           dt_hazard_scalar
#define dt_log_hazard       dt_log_hazard_scalar
#define dt_survival         dt_survival_scalar
#define dt_expected         dt_expected_scalar
#define dt_kelly_mult       dt_kelly_mult_scalar

/* Batch operations - dispatch to best SIMD */
#if DT_USE_AVX512
    #define dt_update_batch         dt_update_16_avx512
    #define dt_get_hazard_batch     dt_get_hazard_16_avx512
    #define dt_get_log_hazard_batch dt_get_log_hazard_16_avx512
    #define dt_get_all_features_batch dt_get_all_features_16_avx512
    #define dt_kelly_mult_batch     dt_kelly_mult_16_avx512
    #define DT_BATCH_SIZE           16
#elif DT_USE_AVX2
    #define dt_update_batch         dt_update_8_avx2
    #define dt_get_hazard_batch     dt_get_hazard_8_avx2
    #define dt_get_log_hazard_batch dt_get_log_hazard_8_avx2
    #define dt_get_all_features_batch dt_get_all_features_8_avx2
    #define dt_kelly_mult_batch     dt_kelly_mult_8_avx2
    #define DT_BATCH_SIZE           8
#else
    /* Scalar fallback for batch - loop wrapper */
    #define DT_BATCH_SIZE           1
    
    static inline void dt_get_hazard_batch_scalar(const DurationLUT *lut,
                                                   const int32_t *regimes,
                                                   const int32_t *durations,
                                                   float *out,
                                                   int n) {
        for (int i = 0; i < n; i++) {
            out[i] = lut->data[regimes[i]][durations[i]][DT_HAZARD];
        }
    }
    #define dt_get_hazard_batch(lut, r, d, out) dt_get_hazard_batch_scalar(lut, r, d, out, 1)
#endif

/* ═══════════════════════════════════════════════════════════════════════════
   Utility - Stability check
   ═══════════════════════════════════════════════════════════════════════════ */

static inline float dt_stability(const DurationTracker *dt) {
    return 1.0f - dt->lut->data[dt->current_regime][dt->ticks_in_regime][DT_HAZARD];
}

static inline int dt_changed(const DurationTracker *dt) {
    return dt->current_regime != dt->prev_regime;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Initialization & I/O (implementation in .c file)
   ═══════════════════════════════════════════════════════════════════════════ */

void dt_init(DurationTracker *dt, const DurationLUT *lut);
int dt_lut_load(DurationLUT *lut, const char *path);
int dt_lut_save(const DurationLUT *lut, const char *path);
void dt_lut_build(DurationLUT *lut,
                  const float hazard_raw[DT_MAX_REGIMES][DT_MAX_DURATION],
                  const float survival_raw[DT_MAX_REGIMES][DT_MAX_DURATION],
                  const float expected_raw[DT_MAX_REGIMES][DT_MAX_DURATION]);

#endif /* DURATION_TRACKER_FAST_H */