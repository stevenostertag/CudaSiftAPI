#include <math.h>
#include <string.h>
#include "cudaSift.h"

/* Portable 32-byte alignment */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define ALIGN32 _Alignas(32)
#elif defined(_MSC_VER)
#define ALIGN32 __declspec(align(32))
#elif defined(__GNUC__)
#define ALIGN32 __attribute__((aligned(32)))
#else
#define ALIGN32
#endif

/*
 * =========================================================================
 * Architecture-Specific Intrinsics Abstraction
 * =========================================================================
 */

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define USE_NEON

static inline float fast_rsqrtf(float x)
{
    float32x4_t vx = vdupq_n_f32(x);
    float32x4_t est = vrsqrteq_f32(vx);
    float32x4_t half_x = vmulq_f32(vdupq_n_f32(0.5f), vx);
    float32x4_t est_sq = vmulq_f32(est, est);
    float32x4_t nr = vsubq_f32(vdupq_n_f32(1.5f), vmulq_f32(half_x, est_sq));
    return vgetq_lane_f32(vmulq_f32(est, nr), 0);
}
static inline void accumulate_outer(float M[8][8], const float *Y, float w)
{
    float32x4_t y_vec0 = vld1q_f32(&Y[0]);
    float32x4_t y_vec1 = vld1q_f32(&Y[4]);
    for (int r = 0; r < 8; r++)
    {
        float32x4_t yr_w = vdupq_n_f32(Y[r] * w);
        float32x4_t m_row0 = vld1q_f32(&M[r][0]);
        float32x4_t m_row1 = vld1q_f32(&M[r][4]);
        m_row0 = vfmaq_f32(m_row0, y_vec0, yr_w);
        m_row1 = vfmaq_f32(m_row1, y_vec1, yr_w);
        vst1q_f32(&M[r][0], m_row0);
        vst1q_f32(&M[r][4], m_row1);
    }
}
static inline void accumulate_vec(float *X, const float *Y, float s)
{
    float32x4_t x_vec0 = vld1q_f32(&X[0]);
    float32x4_t x_vec1 = vld1q_f32(&X[4]);
    float32x4_t y_vec0 = vld1q_f32(&Y[0]);
    float32x4_t y_vec1 = vld1q_f32(&Y[4]);
    float32x4_t s_vec = vdupq_n_f32(s);
    x_vec0 = vfmaq_f32(x_vec0, y_vec0, s_vec);
    x_vec1 = vfmaq_f32(x_vec1, y_vec1, s_vec);
    vst1q_f32(&X[0], x_vec0);
    vst1q_f32(&X[4], x_vec1);
}
static inline void zero_system(float M[8][8], float X[8])
{
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (int r = 0; r < 8; r++)
    {
        vst1q_f32(&M[r][0], zero);
        vst1q_f32(&M[r][4], zero);
    }
    vst1q_f32(&X[0], zero);
    vst1q_f32(&X[4], zero);
}
static inline void prefetch_point(const SiftPoint *pt)
{
    __builtin_prefetch((const void *)pt, 0, 3);
}

#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define USE_AVX2

static inline float fast_rsqrtf(float x)
{
    __m128 vx = _mm_set_ss(x);
    __m128 est = _mm_rsqrt_ss(vx);
    __m128 half_x = _mm_mul_ss(_mm_set_ss(0.5f), vx);
    __m128 est_sq = _mm_mul_ss(est, est);
    __m128 nr = _mm_sub_ss(_mm_set_ss(1.5f), _mm_mul_ss(half_x, est_sq));
    return _mm_cvtss_f32(_mm_mul_ss(est, nr));
}
static inline void accumulate_outer(float M[8][8], const float *Y, float w)
{
    __m256 y_vec = _mm256_load_ps(Y);
    for (int r = 0; r < 8; r++)
    {
        __m256 yr_w = _mm256_set1_ps(Y[r] * w);
        __m256 m_row = _mm256_load_ps(&M[r][0]);
        m_row = _mm256_fmadd_ps(y_vec, yr_w, m_row);
        _mm256_store_ps(&M[r][0], m_row);
    }
}
static inline void accumulate_vec(float *X, const float *Y, float s)
{
    __m256 x_vec = _mm256_load_ps(X);
    __m256 y_vec = _mm256_load_ps(Y);
    __m256 s_vec = _mm256_set1_ps(s);
    x_vec = _mm256_fmadd_ps(y_vec, s_vec, x_vec);
    _mm256_store_ps(X, x_vec);
}
static inline void zero_system(float M[8][8], float X[8])
{
    __m256 zero = _mm256_setzero_ps();
    for (int r = 0; r < 8; r++)
        _mm256_store_ps(&M[r][0], zero);
    _mm256_store_ps(X, zero);
}
static inline void prefetch_point(const SiftPoint *pt)
{
    _mm_prefetch((const char *)pt, _MM_HINT_T0);
}
#endif

/*
 * =========================================================================
 * Shared Algorithm Logic
 * =========================================================================
 */

static int solve_cholesky_8x8(float M[8][8], float b[8])
{
    float L[8][8];
    memset(L, 0, sizeof(L));

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < j; k++)
                sum += L[i][k] * L[j][k];
            if (i == j)
            {
                float val = M[i][i] - sum;
                if (val <= 0.0f)
                    return -1;
                L[i][j] = sqrtf(val);
            }
            else
            {
                L[i][j] = (M[i][j] - sum) / L[j][j];
            }
        }
    }
    for (int i = 0; i < 8; i++)
    {
        float sum = 0.0f;
        for (int k = 0; k < i; k++)
            sum += L[i][k] * b[k];
        b[i] = (b[i] - sum) / L[i][i];
    }
    for (int i = 7; i >= 0; i--)
    {
        float sum = 0.0f;
        for (int k = i + 1; k < 8; k++)
            sum += L[k][i] * b[k];
        b[i] = (b[i] - sum) / L[i][i];
    }
    return 0;
}

int ImproveHomography(SiftData *data, float *homography, int numLoops,
                      float minScore, float maxAmbiguity, float thresh)
{
    if (data->h_data == NULL)
        return 0;

    SiftPoint *mpts = data->h_data;
    float limit = thresh * thresh;
    int numPts = data->numPts;
    float A[8];
    float inv_h8 = 1.0f / homography[8];

    for (int i = 0; i < 8; i++)
        A[i] = homography[i] * inv_h8;

    for (int loop = 0; loop < numLoops; loop++)
    {
        ALIGN32 float M[8][8];
        ALIGN32 float X[8];
        ALIGN32 float Y[8];

        zero_system(M, X);

        for (int i = 0; i < numPts; i++)
        {
            if (i + 1 < numPts)
                prefetch_point(&mpts[i + 1]);

            SiftPoint *pt = &mpts[i];
            if (pt->score < minScore || pt->ambiguity > maxAmbiguity)
                continue;

            float xp = pt->xpos;
            float yp = pt->ypos;
            float mx = pt->match_xpos;
            float my = pt->match_ypos;
            float den = A[6] * xp + A[7] * yp + 1.0f;
            float inv_den = 1.0f / den;
            float dx = (A[0] * xp + A[1] * yp + A[2]) * inv_den - mx;
            float dy = (A[3] * xp + A[4] * yp + A[5]) * inv_den - my;
            float err_sq = dx * dx + dy * dy;

            float wei = (err_sq <= limit) ? 1.0f : 0.0f;

            /* --- x-equation contribution --- */
            Y[0] = xp;
            Y[1] = yp;
            Y[2] = 1.0f;
            Y[3] = 0.0f;
            Y[4] = 0.0f;
            Y[5] = 0.0f;
            Y[6] = -xp * mx;
            Y[7] = -yp * mx;

            accumulate_outer(M, Y, wei);
            accumulate_vec(X, Y, mx * wei);

            /* --- y-equation contribution --- */
            Y[0] = 0.0f;
            Y[1] = 0.0f;
            Y[2] = 0.0f;
            Y[3] = xp;
            Y[4] = yp;
            Y[5] = 1.0f;
            Y[6] = -xp * my;
            Y[7] = -yp * my;

            accumulate_outer(M, Y, wei);
            accumulate_vec(X, Y, my * wei);
        }

        if (solve_cholesky_8x8(M, X) != 0)
            break;
        for (int i = 0; i < 8; i++)
            A[i] = X[i];
    }

    int numfit = 0;
    for (int i = 0; i < numPts; i++)
    {
        SiftPoint *pt = &mpts[i];
        float den = A[6] * pt->xpos + A[7] * pt->ypos + 1.0f;
        float dx = (A[0] * pt->xpos + A[1] * pt->ypos + A[2]) / den - pt->match_xpos;
        float dy = (A[3] * pt->xpos + A[4] * pt->ypos + A[5]) / den - pt->match_ypos;
        float err = dx * dx + dy * dy;

        if (err < limit)
            numfit++;
        pt->match_error = sqrtf(err);
    }

    for (int i = 0; i < 8; i++)
        homography[i] = A[i];
    homography[8] = 1.0f;

    return numfit;
}
