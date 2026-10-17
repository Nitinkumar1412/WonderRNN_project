/* One inference step of the wnn network (see src/wnn/models.py), compiled on the
 * scoring machine by solution.py:  gcc -O3 -march=native -shared -fPIC kernel.c -lm
 *
 * Weights are float32, stored input-major ([in][out]). Matrix-vector products keep 32
 * outputs in vector registers across the whole input, so each weight is read once.
 * Gates use an 8-lane polynomial exp (Cephes expf, ~2 ulp) instead of scalar libm calls.
 * Results are deterministic for a given binary.
 *
 * ctx[] (all pointers, set up once by solution.py):
 *   0 int   cfg[12]   kind(0 gru,1 lrnn,2 hybrid) n_raw diff d gh gl nb S m reset n_out wf16
 *   1 float fcfg[2]   clip, out_bound
 *   2 int   off[103]  float offset of each tensor in W (-1 = absent), order in OFFSETS below
 *   3 float W[]       all weights
 *   4 float hs[]      recurrent state: GRU layers (gl*gh) then blocks (nb*S)
 *   5 float prev[]    previous standardised row (for dx)
 *   6 float sc[]      scratch (see SCRATCH)
 *   7 float raw[]     this row's 112 features (already rounded like the training store)
 *   8 float out[]     n_out outputs
 *   9 u16   Wh[]      weight matrices as IEEE float16 when cfg[11] = 1 (their offsets then index Wh)
 */
#include <math.h>
#include <string.h>

#define MAX_GRU 4
#define MAX_BLK 8
/* OFFSETS: 0 mean, 1 std, 2 proj_w, 3 proj_b,
 *          4 + 4*l + {0 wi, 1 bi, 2 wh, 3 bh}                         l < MAX_GRU
 *          20 + 10*k + {0 n1, 1 win, 2 bin, 3 wout, 4 bout, 5 n2, 6 w1, 7 b1, 8 w2, 9 b2}   k < MAX_BLK
 *          100 norm_f, 101 head_w, 102 head_b */

/* 8 floats; aligned(4) allows unaligned loads. GCC lowers it to AVX/AVX2 (-march=native) or SSE pairs. */
typedef float v8 __attribute__((vector_size(32), aligned(4)));
typedef int v8i __attribute__((vector_size(32), aligned(4)));

static inline v8 vsplat(float c) { return (v8){c, c, c, c, c, c, c, c}; }
static inline v8 vsel(v8i m, v8 a, v8 b) { return (v8)((m & (v8i)a) | (~m & (v8i)b)); }

/* Cephes expf: exp(x) = 2^n * p(r), |r| <= ln2/2 */
static inline v8 vexp(v8 x) {
    x = vsel(x > vsplat(88.3762626647949f), vsplat(88.3762626647949f), x);
    x = vsel(x < vsplat(-87.3365447504f), vsplat(-87.3365447504f), x);
    v8 fx = x * vsplat(1.44269504088896341f) + vsplat(0.5f);
    v8i ti = __builtin_convertvector(fx, v8i);              /* trunc toward zero */
    v8 t = __builtin_convertvector(ti, v8);
    v8 n = vsel(t > fx, t - vsplat(1.0f), t);              /* floor */
    v8 r = x - n * vsplat(0.693359375f) - n * vsplat(-2.12194440e-4f);
    v8 r2 = r * r;
    v8 p = vsplat(1.9875691500E-4f);
    p = p * r + vsplat(1.3981999507E-3f);
    p = p * r + vsplat(8.3334519073E-3f);
    p = p * r + vsplat(4.1665795894E-2f);
    p = p * r + vsplat(1.6666665459E-1f);
    p = p * r + vsplat(5.0000001201E-1f);
    p = p * r2 + r + vsplat(1.0f);
    v8i e = (__builtin_convertvector(n, v8i) + 127) << 23;
    return p * (v8)e;
}
static inline v8 vsigm(v8 x) { return vsplat(1.0f) / (vsplat(1.0f) + vexp(-x)); }
static inline v8 vtanh(v8 x) { return vsplat(2.0f) / (vsplat(1.0f) + vexp(vsplat(-2.0f) * x)) - vsplat(1.0f); }
static inline v8 vgelu(v8 x) {
    return vsplat(0.5f) * x * (vsplat(1.0f) + vtanh(vsplat(0.7978845608028654f) * (x + vsplat(0.044715f) * x * x * x)));
}

/* n is padded up to a multiple of 8 by the callers' scratch layout (tails read/write spare lanes) */
#define VLOOP(n) for (int _j = 0; _j < (n); _j += 8)
#define LD(p) (*(const v8 *)((p) + _j))
#define ST(p, v) (*(v8 *)((p) + _j) = (v))

/* y = x W + b, W input-major [n_in][n_out]. Outputs in register blocks of 32 over the whole input. */
static void matvec(const float *restrict x, int n_in, const float *restrict W, const float *restrict b,
                   int n_out, float *restrict y) {
    const size_t ld = (size_t)n_out;
    int o = 0;
    for (; o + 32 <= n_out; o += 32) {
        v8 a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0};
        if (b) {
            a0 = *(const v8 *)(b + o); a1 = *(const v8 *)(b + o + 8);
            a2 = *(const v8 *)(b + o + 16); a3 = *(const v8 *)(b + o + 24);
        }
        const float *w = W + o;
        for (int i = 0; i < n_in; i++, w += ld) {
            const float xi = x[i];
            a0 += xi * *(const v8 *)(w);
            a1 += xi * *(const v8 *)(w + 8);
            a2 += xi * *(const v8 *)(w + 16);
            a3 += xi * *(const v8 *)(w + 24);
        }
        *(v8 *)(y + o) = a0; *(v8 *)(y + o + 8) = a1;
        *(v8 *)(y + o + 16) = a2; *(v8 *)(y + o + 24) = a3;
    }
    for (; o + 8 <= n_out; o += 8) {
        v8 a0 = {0};
        if (b) a0 = *(const v8 *)(b + o);
        const float *w = W + o;
        for (int i = 0; i < n_in; i++, w += ld) a0 += x[i] * *(const v8 *)(w);
        *(v8 *)(y + o) = a0;
    }
    for (; o < n_out; o++) {
        float a = b ? b[o] : 0.0f;
        for (int i = 0; i < n_in; i++) a += x[i] * W[(size_t)i * ld + o];
        y[o] = a;
    }
}

/* float16 weights: half the bytes per step (the step is memory-bound); F16C converts 8 at a time */
static inline float h2f(unsigned short h) {
    unsigned int s = (unsigned int)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, u;
    if (e == 0) {
        if (m == 0) u = s;
        else {
            e = 1;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3ffu;
            u = s | ((e + 112u) << 23) | (m << 13);
        }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112u) << 23) | (m << 13);
    float f;
    memcpy(&f, &u, 4);
    return f;
}
#if defined(__F16C__)
#include <immintrin.h>
static inline v8 ldh(const unsigned short *p) { return (v8)_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p)); }
#else
static inline v8 ldh(const unsigned short *p) {
    v8 r;
    for (int i = 0; i < 8; i++) r[i] = h2f(p[i]);
    return r;
}
#endif

static void matvec_h(const float *restrict x, int n_in, const unsigned short *restrict W, const float *restrict b,
                     int n_out, float *restrict y) {
    const size_t ld = (size_t)n_out;
    int o = 0;
    for (; o + 32 <= n_out; o += 32) {
        v8 a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0};
        if (b) {
            a0 = *(const v8 *)(b + o); a1 = *(const v8 *)(b + o + 8);
            a2 = *(const v8 *)(b + o + 16); a3 = *(const v8 *)(b + o + 24);
        }
        const unsigned short *w = W + o;
        for (int i = 0; i < n_in; i++, w += ld) {
            const float xi = x[i];
            a0 += xi * ldh(w);
            a1 += xi * ldh(w + 8);
            a2 += xi * ldh(w + 16);
            a3 += xi * ldh(w + 24);
        }
        *(v8 *)(y + o) = a0; *(v8 *)(y + o + 8) = a1;
        *(v8 *)(y + o + 16) = a2; *(v8 *)(y + o + 24) = a3;
    }
    for (; o + 8 <= n_out; o += 8) {
        v8 a0 = {0};
        if (b) a0 = *(const v8 *)(b + o);
        const unsigned short *w = W + o;
        for (int i = 0; i < n_in; i++, w += ld) a0 += x[i] * ldh(w);
        *(v8 *)(y + o) = a0;
    }
    for (; o < n_out; o++) {
        float a = b ? b[o] : 0.0f;
        for (int i = 0; i < n_in; i++) a += x[i] * h2f(W[(size_t)i * ld + o]);
        y[o] = a;
    }
}

static void rmsnorm(const float *x, int d, const float *g, float *y) {
    float ss = 0.0f;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    const float r = 1.0f / sqrtf(ss / (float)d + 1e-6f);
    for (int i = 0; i < d; i++) y[i] = x[i] * r * g[i];
}

static inline int up8(int n) { return (n + 7) & ~7; }

void wnn_step(void **ctx, int first) {
    const int *c = (const int *)ctx[0];
    const float *fc = (const float *)ctx[1];
    const int *off = (const int *)ctx[2];
    const float *W = (const float *)ctx[3];
    float *hs = (float *)ctx[4], *prev = (float *)ctx[5], *sc = (float *)ctx[6];
    const float *raw = (const float *)ctx[7];
    float *out = (float *)ctx[8];
    const int kind = c[0], nr = c[1], diff = c[2], d = c[3], gh = c[4], gl = c[5], nb = c[6];
    const int S = c[7], m = c[8], n_out = c[10];
    const float clip = fc[0], bound = fc[1];
    const int wf16 = c[11];
    const unsigned short *Wh = (const unsigned short *)ctx[9];
#define P(k) (off[(k)] >= 0 ? W + off[(k)] : (const float *)0)
/* weight matrix k (offset into Wh when float16) */
#define MV(xx, nin, k, bias, nout, yy)     (wf16 ? matvec_h((xx), (nin), Wh + off[(k)], (bias), (nout), (yy)) : matvec((xx), (nin), P(k), (bias), (nout), (yy)))

    /* SCRATCH: f[2nr] u[w] v[w] a[A] b[A] t[A], every buffer padded to a multiple of 8 */
    int wmax = up8(d > gh ? d : gh);
    int amax = 3 * gh;
    if (2 * S > amax) amax = 2 * S;
    if (m > amax) amax = m;
    if (wmax > amax) amax = wmax;
    amax = up8(amax) + 8;
    float *f = sc;
    float *u = f + up8(2 * nr);
    float *v = u + wmax;
    float *a = v + wmax;
    float *b = a + amax;
    float *t = b + amax;

    const float *mean = P(0), *std = P(1);
    for (int i = 0; i < nr; i++) {
        float x = (raw[i] - mean[i]) / std[i];
        if (x > clip) x = clip;
        if (x < -clip) x = -clip;
        f[i] = x;
        if (diff) f[nr + i] = first ? 0.0f : x - prev[i];
        prev[i] = x;
    }
    const int n_in = diff ? 2 * nr : nr;
    MV(f, n_in, 2, P(3), d, a);
    VLOOP(d) ST(u, vgelu(LD(a)));
    int width = d;

    if (kind == 0 || kind == 2) {
        const float *inp = u;
        int in_dim = d;
        for (int l = 0; l < gl; l++) {
            float *h = hs + l * gh;
            MV(inp, in_dim, 4 + 4 * l, P(5 + 4 * l), 3 * gh, a);
            MV(h, gh, 6 + 4 * l, P(7 + 4 * l), 3 * gh, b);
            /* gh is a multiple of 8 in every preset; the scalar path covers the rest */
            int j = 0;
            for (; j + 8 <= gh; j += 8) {
                v8 r = vsigm(*(v8 *)(a + j) + *(v8 *)(b + j));
                v8 z = vsigm(*(v8 *)(a + gh + j) + *(v8 *)(b + gh + j));
                v8 n = vtanh(*(v8 *)(a + 2 * gh + j) + r * *(v8 *)(b + 2 * gh + j));
                *(v8 *)(h + j) = (vsplat(1.0f) - z) * n + z * *(v8 *)(h + j);
            }
            for (; j < gh; j++) {
                const float r = 1.0f / (1.0f + expf(-(a[j] + b[j])));
                const float z = 1.0f / (1.0f + expf(-(a[gh + j] + b[gh + j])));
                const float n = tanhf(a[2 * gh + j] + r * b[2 * gh + j]);
                h[j] = (1.0f - z) * n + z * h[j];
            }
            inp = h;
            in_dim = gh;
        }
        memcpy(u, inp, sizeof(float) * (size_t)gh);
        width = gh;
    }
    if (kind == 1 || kind == 2) {
        float *hb = hs + gl * gh;
        for (int k = 0; k < nb; k++) {
            const int o = 20 + 10 * k;
            float *h = hb + k * S;
            rmsnorm(u, d, P(o + 0), v);
            MV(v, d, o + 1, P(o + 2), 2 * S, a);
            int j = 0;
            for (; j + 8 <= S; j += 8) {
                v8 z = vsigm(*(v8 *)(a + j));
                v8 bb = *(v8 *)(a + S + j);
                v8 g = vsel(bb >= vsplat(0.0f), bb + vsplat(0.5f), vsigm(bb));
                *(v8 *)(h + j) = (vsplat(1.0f) - z) * *(v8 *)(h + j) + z * g;
            }
            for (; j < S; j++) {
                const float z = 1.0f / (1.0f + expf(-a[j]));
                const float bb = a[S + j];
                const float g = bb >= 0.0f ? bb + 0.5f : 1.0f / (1.0f + expf(-bb));
                h[j] = (1.0f - z) * h[j] + z * g;
            }
            MV(h, S, o + 3, P(o + 4), d, b);
            for (int q = 0; q < d; q++) u[q] += b[q];
            if (m > 0) {
                rmsnorm(u, d, P(o + 5), v);
                MV(v, d, o + 6, P(o + 7), m, t);
                VLOOP(m) ST(t, vgelu(LD(t)));
                MV(t, m, o + 8, P(o + 9), d, b);
                for (int q = 0; q < d; q++) u[q] += b[q];
            }
        }
        rmsnorm(u, d, P(100), v);
        memcpy(u, v, sizeof(float) * (size_t)d);
        width = d;
    }
    MV(u, width, 101, P(102), n_out, out);
    if (bound > 0.0f)
        for (int j = 0; j < n_out; j++) out[j] = bound * tanhf(out[j] / bound);
#undef MV
#undef P
}
