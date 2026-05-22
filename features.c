/*
============================================================
WORK-IN-PROGRESS C PORT
============================================================

ONLY high-confidence sections have been ported to C.

The following sections are STILL PYTHON and intentionally
left unresolved because they require architectural decisions:

    - MODWT / SWT wavelets
    - AR Burg root analysis

Those sections are clearly marked below.

Goal:
    - deterministic
    - embedded-friendly
    - easy to expand later
    - no dynamic Python-style abstractions

============================================================
*/


#include <math.h>
#include <stdlib.h>
#include <string.h>


/*
============================================================
CONFIG
============================================================
*/

#define EPSILON 1e-12f

#define NUM_BANDS 3

#define NUM_TIME_FEATURES 6
#define NUM_FREQ_FEATURES 5
#define NUM_STABILITY_FEATURES 1

#define TOTAL_FEATURES \
    (NUM_TIME_FEATURES + \
     NUM_FREQ_FEATURES + \
     NUM_STABILITY_FEATURES)


/*
============================================================
FEATURE STRUCT
============================================================
*/

typedef struct {

    /*
    ----------------------------
    Time domain
    ----------------------------
    */

    float mean;
    float variance;
    float rms;
    float skew;
    float kurtosis;
    float zero_crossing_rate;

    /*
    ----------------------------
    Frequency domain
    ----------------------------
    */

    float band_energy[NUM_BANDS];

    float spectral_centroid;

    float spectral_flatness;

    /*
    ----------------------------
    Stability
    ----------------------------
    */

    float mean_squared_diff;

} FeatureVector;


/*
============================================================
UTILITY FUNCTIONS
============================================================
*/


static float compute_mean(
    const float* x,
    int N
) {
    float sum = 0.0f;

    for (int i = 0; i < N; i++) {
        sum += x[i];
    }

    return sum / (float)N;
}


static float compute_variance(
    const float* x,
    int N,
    float mean
) {
    float sum = 0.0f;

    for (int i = 0; i < N; i++) {

        float d = x[i] - mean;

        sum += d * d;
    }

    return sum / (float)N;
}


static float compute_rms(
    const float* x,
    int N
) {
    float sum = 0.0f;

    for (int i = 0; i < N; i++) {
        sum += x[i] * x[i];
    }

    return sqrtf(sum / (float)N);
}


/*
============================================================
SKEWNESS
High-confidence implementation.
============================================================
*/

static float compute_skew(
    const float* x,
    int N,
    float mean,
    float variance
) {
    float stddev = sqrtf(variance + EPSILON);

    float accum = 0.0f;

    for (int i = 0; i < N; i++) {

        float z = (x[i] - mean) / stddev;

        accum += z * z * z;
    }

    return accum / (float)N;
}


/*
============================================================
KURTOSIS
Matches scipy default style approximately
(excess kurtosis).
============================================================
*/

static float compute_kurtosis(
    const float* x,
    int N,
    float mean,
    float variance
) {
    float stddev = sqrtf(variance + EPSILON);

    float accum = 0.0f;

    for (int i = 0; i < N; i++) {

        float z = (x[i] - mean) / stddev;

        accum += z * z * z * z;
    }

    return (accum / (float)N) - 3.0f;
}


/*
============================================================
ZERO CROSSING RATE
============================================================
*/

static float compute_zero_crossing_rate(
    const float* x,
    int N
) {
    int crossings = 0;

    for (int i = 1; i < N; i++) {

        if (
            (x[i - 1] >= 0.0f && x[i] < 0.0f) ||
            (x[i - 1] < 0.0f && x[i] >= 0.0f)
        ) {
            crossings++;
        }
    }

    return (float)crossings / (float)(N - 1);
}


/*
============================================================
TIME DOMAIN FEATURES
============================================================
*/

void extract_time_features(
    const float* x,
    int N,
    FeatureVector* out
) {

    float mean = compute_mean(x, N);

    float variance = compute_variance(
        x,
        N,
        mean
    );

    out->mean = mean;

    out->variance = variance;

    out->rms = compute_rms(x, N);

    out->skew = compute_skew(
        x,
        N,
        mean,
        variance
    );

    out->kurtosis = compute_kurtosis(
        x,
        N,
        mean,
        variance
    );

    out->zero_crossing_rate =
        compute_zero_crossing_rate(x, N);
}


/*
============================================================
STABILITY FEATURE
============================================================
*/

void extract_stability_feature(
    const float* x,
    int N,
    FeatureVector* out
) {

    float accum = 0.0f;

    for (int i = 1; i < N; i++) {

        float d = x[i] - x[i - 1];

        accum += d * d;
    }

    out->mean_squared_diff =
        accum / (float)(N - 1);
}


/*
============================================================
FFT SECTION
============================================================

HIGH CONFIDENCE ARCHITECTURE

BUT:

Actual FFT implementation intentionally omitted.

You should later plug in one of:

    - KissFFT
    - CMSIS-DSP
    - FFTW

Expected inputs:
    real[]
    imag[]

after FFT execution.

============================================================
*/


void extract_frequency_features(
    const float* real,
    const float* imag,
    int fft_bins,
    float fs,
    FeatureVector* out
) {

    /*
    Frequency bands:
    */

    float band_edges[NUM_BANDS][2] = {
        {0.0f, 10.0f},
        {10.0f, 100.0f},
        {100.0f, 500.0f}
    };

    float total_power = 0.0f;

    float centroid_numerator = 0.0f;

    float flatness_log_sum = 0.0f;

    for (int i = 0; i < NUM_BANDS; i++) {
        out->band_energy[i] = 0.0f;
    }

    /*
    Compute PSD
    */

    for (int k = 0; k < fft_bins; k++) {

        float freq =
            ((float)k * fs) /
            (2.0f * (float)(fft_bins - 1));

        float power =
            real[k] * real[k] +
            imag[k] * imag[k];

        total_power += power;

        centroid_numerator +=
            freq * power;

        flatness_log_sum +=
            logf(power + EPSILON);

        /*
        Band energies
        */

        for (int b = 0; b < NUM_BANDS; b++) {

            float lo = band_edges[b][0];
            float hi = band_edges[b][1];

            if (freq >= lo && freq < hi) {

                out->band_energy[b] += power;
            }
        }
    }

    total_power += EPSILON;

    /*
    Normalize band energies
    */

    for (int b = 0; b < NUM_BANDS; b++) {

        out->band_energy[b] /=
            total_power;
    }

    /*
    Spectral centroid
    */

    out->spectral_centroid =
        centroid_numerator /
        total_power;

    /*
    Spectral flatness
    */

    float geometric_mean =
        expf(flatness_log_sum /
        (float)fft_bins);

    float arithmetic_mean =
        total_power /
        (float)fft_bins;

    out->spectral_flatness =
        geometric_mean /
        (arithmetic_mean + EPSILON);
}


/*
============================================================
MAIN EXTRACTION PIPELINE
============================================================
*/

void extract_feature_vector(
    const float* signal,
    int N,

    /*
    FFT outputs
    */

    const float* fft_real,
    const float* fft_imag,
    int fft_bins,

    float fs,

    FeatureVector* out
) {

    /*
    Time domain
    */

    extract_time_features(
        signal,
        N,
        out
    );

    /*
    Frequency domain
    */

    extract_frequency_features(
        fft_real,
        fft_imag,
        fft_bins,
        fs,
        out
    );

    /*
    Stability
    */

    extract_stability_feature(
        signal,
        N,
        out
    );
}


/*
============================================================
TODO / DISCUSS
============================================================

The following Python code is intentionally left
unported because implementation strategy still
needs discussion.

------------------------------------------------------------
1. MODWT / SWT WAVELETS
------------------------------------------------------------

Current Python:

    coeffs = pywt.swt(window, "db4", level=4)

Possible future directions:

    A) Remove entirely
    B) Use wavelib
    C) Handwritten db4 implementation

------------------------------------------------------------
2. AR BURG FEATURES
------------------------------------------------------------

Current Python:

    arburg(...)
    np.roots(...)

Main issue:
    polynomial root solving on embedded systems.

Likely future direction:

    - keep AR coefficients only
    - remove roots
    - use LPC features instead

============================================================
*/
