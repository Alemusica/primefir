// primefir_tilde.cpp — FIR windowed-sinc con offset aperiodici (primi, φ, √2, ρ, π, e)
// + finestre: hann/hamming/blackman/blackmanharris/nuttall/kaiser (radiali)
// + interpolazione frazionaria: off / linear / lagrange4 / catmullrom / farrow3 / farrow5
// + stereo 2in/2out
// © 2025 — MIT License. Compilare C++17. Max SDK 8+.

extern "C" {
  #include "ext.h"
  #include "ext_obex.h"
  #include "z_dsp.h"
}
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E  2.71828182845904523536
#endif

// =====================  COSTANTI DSP  =====================
static constexpr int     kRingSize   = 131072;              // potenza di 2
static constexpr int     kRingMask   = kRingSize - 1;
static constexpr int     kMaxWindow  = 4096;                // profondità FIR massima
static constexpr double  kTiny       = 1.0e-30;
static constexpr int     kPrimeTableSize = int(kMaxWindow * 2.1) + 64;

static constexpr double kPhi     = 1.6180339887498948482;
static constexpr double kSqrt2   = 1.4142135623730950488;
static constexpr double kPlastic = 1.3247179572447458000;

// =====================  ENUM MODALITÀ  =====================
enum class seq_mode : int {
  linear = 0, prime, phi, sqrt2, plastic, pi, e,
  prime_phi_index,   // p_{ floor(n*phi) }
  prime_phi_scaled   // floor(phi * p_n)
};

// Finestre (radiali: picco al centro d=0 → bordo d=w-1)
enum class winshape : int {
  hann = 0, hamming, blackman, blackmanharris, nuttall, kaiser
};

// Interpolazione
// 0=off, 1=linear, 2=lagrange4 (causale), 3=catmullrom (Keys a=-0.5),
// 4=farrow3 (cubic, struttura Farrow), 5=farrow5 (quintic, struttura Farrow)
enum class interp_mode : int { off = 0, linear, lagrange4, catmullrom, farrow3, farrow5 };

// =====================  STRUTTURA OGGETTO  =====================
typedef struct _primefir {
  t_pxobject  ob;

  // Parametri esposti
  double      param_freq;       // [0..1]
  double      param_window;     // [0..1]
  long        param_mode;       // seq_mode
  char        param_normalize;  // 0/1
  char        param_gaincomp;   // 0/1
  char        param_linphase;   // 0/1, abilita kernel simmetrico
  long        param_interp;     // interp_mode
  long        param_winshape;   // winshape
  double      param_kaiser_beta;// Kaiser β
  double      param_keys_a;     // Keys 'a' (Catmull-Rom = -0.5)

  // Stato DSP
  double      sr;
  double      ringL[kRingSize];
  double      ringR[kRingSize];
  uint32_t    write_idx;

  // Kernel (distanza-indicizzato)
  int         window;                  // numero tap
  int         middle;                  // non usato in somma (compat.)
  double      fir[kMaxWindow + 1];     // fir[d], d=0..w-1 (d=0 non sommato)
  uint32_t    ioffs[kMaxWindow + 1];   // parte intera del ritardo per d
  double      ffrac[kMaxWindow + 1];   // frazione [0,1) per d
  double      post_scale;
  bool        dirty;
  uint32_t    latency;                 // L = max(D) + 1
  double      fir0;                    // tap centrale (d=0)

  double      w_lin_fwd [kMaxWindow + 1][2];
  double      w_lag4_fwd[kMaxWindow + 1][4];
  double      w_keys_fwd[kMaxWindow + 1][4];
  double      w_far3_fwd[kMaxWindow + 1][4];
  double      w_far5_fwd[kMaxWindow + 1][6];

  // Primi
  int         primes[kPrimeTableSize]; // primes[1..primes_count]
  int         primes_count;
  bool        primes_ready;
} t_primefir;

// =====================  PROTOTIPI  =====================
void*       primefir_new(t_symbol* s, long argc, t_atom* argv);
void        primefir_free(t_primefir* x);
void        primefir_assist(t_primefir* x, void* b, long m, long a, char* s);

void        primefir_clear(t_primefir* x);
void        primefir_dsp64(t_primefir* x, t_object* dsp64, short* count, double sr, long n, long flags);
void        primefir_perform64(t_primefir* x, t_object* dsp64, double** ins, long numins,
                               double** outs, long numouts, long sampleframes, long flags, void* userparam);

// Attributi
t_max_err   primefir_attr_set_freq(t_primefir* x, void* attr, long ac, t_atom* av);
t_max_err   primefir_attr_set_window(t_primefir* x, void* attr, long ac, t_atom* av);
t_max_err   primefir_attr_set_mode(t_primefir* x, void* attr, long ac, t_atom* av);
t_max_err   primefir_attr_set_normalize(t_primefir* x, void* attr, long ac, t_atom* av);
t_max_err   primefir_attr_set_gaincomp(t_primefir* x, void* attr, long ac, t_atom* av);
t_max_err   primefir_attr_set_linphase(t_primefir* x, void* attr, long ac, t_atom* av);
t_max_err   primefir_attr_set_interp(t_primefir* x, void* attr, long ac, t_atom* av);
t_max_err   primefir_attr_set_winshape(t_primefir* x, void* attr, long ac, t_atom* av);
t_max_err   primefir_attr_set_kaiser_beta(t_primefir* x, void* attr, long ac, t_atom* av);
t_max_err   primefir_attr_set_keys_a(t_primefir* x, void* attr, long ac, t_atom* av);

// Utilità
static inline double clamp01(double v) { return (v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v)); }
void        primefir_make_primes(t_primefir* x, int count_needed);

static int  prime_upper_bound_from_count(int count_needed);
void        primefir_update_kernel(t_primefir* x);

// Bessel I0 (Kaiser)
static inline double i0_approx(double x) {
  double ax = std::abs(x);
  if (ax <= 3.75) {
    double t = ax / 3.75;
    double t2 = t * t;
    return 1.0 + t2*(3.5156229 + t2*(3.0899424 + t2*(1.2067492 + t2*(0.2659732 + t2*(0.0360768 + t2*0.0045813)))));
  } else {
    double t = 3.75 / ax;
    double e = std::exp(ax) / std::sqrt(ax);
    double p = 0.39894228 + t*(0.01328592 + t*(0.00225319 + t*(-0.00157565 + t*(0.00916281 + t*(-0.02057706 + t*(0.02635537 + t*(-0.01647633 + t*0.00392377)))))));
    return e * p;
  }
}

// =====================  REGISTRAZIONE CLASSE  =====================
static t_class* s_primefir_class = nullptr;

extern "C" int C74_EXPORT main(void)
{
  t_class* c = class_new("primefir~",
                         (method)primefir_new,
                         (method)primefir_free,
                         (long)sizeof(t_primefir),
                         0L, A_GIMME, 0);

  class_addmethod(c, (method)primefir_assist, "assist", A_CANT, 0);
  class_addmethod(c, (method)primefir_clear,  "clear",  0);
  class_addmethod(c, (method)primefir_dsp64,  "dsp64",  A_CANT, 0);
  class_dspinit(c);

  // ===== Attributi =====
  CLASS_ATTR_DOUBLE(c,  "freq",      0, t_primefir, param_freq);
  CLASS_ATTR_ACCESSORS(c, "freq", NULL, primefir_attr_set_freq);
  CLASS_ATTR_FILTER_CLIP(c, "freq", 0.0, 1.0);
  CLASS_ATTR_LABEL(c,  "freq",      0, "Freq (0..1)");

  CLASS_ATTR_DOUBLE(c,  "window",    0, t_primefir, param_window);
  CLASS_ATTR_ACCESSORS(c, "window", NULL, primefir_attr_set_window);
  CLASS_ATTR_FILTER_CLIP(c, "window", 0.0, 1.0);
  CLASS_ATTR_LABEL(c,  "window",    0, "Window (0..1)");

  CLASS_ATTR_LONG(c,    "mode",      0, t_primefir, param_mode);
  CLASS_ATTR_ACCESSORS(c, "mode", NULL, primefir_attr_set_mode);
  CLASS_ATTR_LABEL(c,   "mode",      0, "Sequence Mode");

  CLASS_ATTR_CHAR(c,    "normalize", 0, t_primefir, param_normalize);
  CLASS_ATTR_ACCESSORS(c, "normalize", NULL, primefir_attr_set_normalize);
  CLASS_ATTR_STYLE_LABEL(c, "normalize", 0, "onoff", "Normalize DC (0/1)");

  CLASS_ATTR_CHAR(c,    "gaincomp",  0, t_primefir, param_gaincomp);
  CLASS_ATTR_ACCESSORS(c, "gaincomp", NULL, primefir_attr_set_gaincomp);
  CLASS_ATTR_STYLE_LABEL(c, "gaincomp",  0, "onoff", "Gain Compensation sqrt(freq) (0/1)");

  CLASS_ATTR_CHAR(c,    "linphase",  0, t_primefir, param_linphase);
  CLASS_ATTR_STYLE_LABEL(c, "linphase", 0, "onoff", "Linear-Phase (symmetric)");
  CLASS_ATTR_ACCESSORS(c, "linphase", NULL, primefir_attr_set_linphase);

  CLASS_ATTR_LONG(c,    "interp",    0, t_primefir, param_interp);
  CLASS_ATTR_ACCESSORS(c, "interp", NULL, primefir_attr_set_interp);
  CLASS_ATTR_STYLE_LABEL(c, "interp", 0, "enumindex", "Interpolation (off/linear/lagrange4/catmullrom/farrow3/farrow5)");
#ifdef CLASS_ATTR_ENUMINDEX
  CLASS_ATTR_ENUMINDEX(c, "interp", 0,
                       "off linear lagrange4 catmullrom farrow3 farrow5");
#else
  CLASS_ATTR_ENUM(c, "interp", 0, "off linear lagrange4 catmullrom farrow3 farrow5");
#endif

  CLASS_ATTR_LONG(c,    "winshape",  0, t_primefir, param_winshape);
  CLASS_ATTR_ACCESSORS(c, "winshape", NULL, primefir_attr_set_winshape);
  CLASS_ATTR_STYLE_LABEL(c, "winshape", 0, "enumindex", "Window (hann/hamming/blackman/blackmanharris/nuttall/kaiser)");
#ifdef CLASS_ATTR_ENUMINDEX
  CLASS_ATTR_ENUMINDEX(c, "winshape", 0,
                       "hann hamming blackman blackmanharris nuttall kaiser");
#else
  CLASS_ATTR_ENUM(c, "winshape", 0, "hann hamming blackman blackmanharris nuttall kaiser");
#endif

  CLASS_ATTR_DOUBLE(c,  "kaiser_beta", 0, t_primefir, param_kaiser_beta);
  CLASS_ATTR_ACCESSORS(c, "kaiser_beta", NULL, primefir_attr_set_kaiser_beta);
  CLASS_ATTR_LABEL(c,  "kaiser_beta", 0, "Kaiser beta");

  CLASS_ATTR_DOUBLE(c,  "keys_a",    0, t_primefir, param_keys_a);
  CLASS_ATTR_ACCESSORS(c, "keys_a", NULL, primefir_attr_set_keys_a);
  CLASS_ATTR_LABEL(c,  "keys_a",    0, "Keys cubic 'a' (Catmull-Rom = -0.5)");

  class_register(CLASS_BOX, c);
  s_primefir_class = c;
  return 0;
}

// =====================  COSTRUZIONE/ DISTRUZIONE  =====================
void* primefir_new(t_symbol* s, long argc, t_atom* argv)
{
  auto* x = (t_primefir*)object_alloc(s_primefir_class);
  if (!x) return nullptr;

  dsp_setup((t_pxobject*)x, 2);
  outlet_new((t_object*)x, "signal");
  outlet_new((t_object*)x, "signal");

  x->sr              = sys_getsr(); if (x->sr <= 0) x->sr = 44100.0;
  x->param_freq      = 1.0;
  x->param_window    = 1.0;
  x->param_mode      = (long)seq_mode::prime;
  x->param_normalize = 0;
  x->param_gaincomp  = 1;
  x->param_linphase  = 0;

  x->param_interp    = (long)interp_mode::farrow5;    // default: massima qualità
  x->param_winshape  = (long)winshape::blackmanharris;// default: BH 4-term
  x->param_kaiser_beta = 9.5;
  x->param_keys_a    = -0.5;

  std::fill(std::begin(x->ringL), std::end(x->ringL), 0.0);
  std::fill(std::begin(x->ringR), std::end(x->ringR), 0.0);
  x->write_idx     = 0;

  x->window        = 2;
  x->middle        = 1;
  std::fill(std::begin(x->fir),   std::end(x->fir),   0.0);
  std::fill(std::begin(x->ioffs), std::end(x->ioffs), 0u);
  std::fill(std::begin(x->ffrac), std::end(x->ffrac), 0.0);
  x->post_scale    = 1.0;
  x->dirty         = true;
  x->latency       = 0;
  x->fir0          = 0.0;
  std::memset(x->w_lin_fwd,  0, sizeof(x->w_lin_fwd));
  std::memset(x->w_lag4_fwd, 0, sizeof(x->w_lag4_fwd));
  std::memset(x->w_keys_fwd, 0, sizeof(x->w_keys_fwd));
  std::memset(x->w_far3_fwd, 0, sizeof(x->w_far3_fwd));
  std::memset(x->w_far5_fwd, 0, sizeof(x->w_far5_fwd));

  x->primes_ready  = false;
  x->primes_count  = 0;

  attr_args_process(x, (short)argc, argv);
  return x;
}

void primefir_free(t_primefir* x) { dsp_free((t_pxobject*)x); }

void primefir_assist(t_primefir* x, void* b, long m, long a, char* s)
{
  if (m == ASSIST_INLET) snprintf_zero(s, 256, (a==0) ? "In L (segnale)" : "In R (segnale) [opz]");
  else                   snprintf_zero(s, 256, (a==0) ? "Out L (FIR aperiodico)" : "Out R (FIR aperiodico)");
}

// =====================  ATTR SETTERS  =====================
static inline double clamp01(double v);
t_max_err primefir_attr_set_freq(t_primefir* x, void*, long ac, t_atom* av) {
  if (ac && av) { x->param_freq = clamp01(atom_getfloat(av)); x->dirty = true; }
  return MAX_ERR_NONE;
}
t_max_err primefir_attr_set_window(t_primefir* x, void*, long ac, t_atom* av) {
  if (ac && av) { x->param_window = clamp01(atom_getfloat(av)); x->dirty = true; }
  return MAX_ERR_NONE;
}
t_max_err primefir_attr_set_mode(t_primefir* x, void*, long ac, t_atom* av) {
  if (ac && av) {
    if (atom_gettype(av) == A_SYM) {
      const char* s = atom_getsym(av)->s_name;
      if (!std::strcmp(s,"linear")) x->param_mode = (long)seq_mode::linear;
      else if (!std::strcmp(s,"prime")) x->param_mode = (long)seq_mode::prime;
      else if (!std::strcmp(s,"phi")) x->param_mode = (long)seq_mode::phi;
      else if (!std::strcmp(s,"sqrt2")) x->param_mode = (long)seq_mode::sqrt2;
      else if (!std::strcmp(s,"plastic")) x->param_mode = (long)seq_mode::plastic;
      else if (!std::strcmp(s,"pi")) x->param_mode = (long)seq_mode::pi;
      else if (!std::strcmp(s,"e")) x->param_mode = (long)seq_mode::e;
      else if (!std::strcmp(s,"prime_phi_index")) x->param_mode = (long)seq_mode::prime_phi_index;
      else if (!std::strcmp(s,"prime_phi_scaled")) x->param_mode = (long)seq_mode::prime_phi_scaled;
    } else {
      long m = atom_getlong(av); if (m < 0) m = 0; if (m > 8) m = 8; x->param_mode = m;
    }
    x->dirty = true;
  }
  return MAX_ERR_NONE;
}
t_max_err primefir_attr_set_normalize(t_primefir* x, void*, long ac, t_atom* av) {
  if (ac && av) { x->param_normalize = (char)(atom_getlong(av) ? 1 : 0); x->dirty = true; }
  return MAX_ERR_NONE;
}
t_max_err primefir_attr_set_gaincomp(t_primefir* x, void*, long ac, t_atom* av) {
  if (ac && av) { x->param_gaincomp = (char)(atom_getlong(av) ? 1 : 0); x->dirty = true; }
  return MAX_ERR_NONE;
}
t_max_err primefir_attr_set_linphase(t_primefir* x, void*, long ac, t_atom* av) {
  if (ac && av) { x->param_linphase = (char)(atom_getlong(av) ? 1 : 0); x->dirty = true; }
  return MAX_ERR_NONE;
}
t_max_err primefir_attr_set_interp(t_primefir* x, void*, long ac, t_atom* av) {
  if (ac && av) {
    if (atom_gettype(av) == A_SYM) {
      const char* s = atom_getsym(av)->s_name;
      if (!std::strcmp(s,"off")) x->param_interp = (long)interp_mode::off;
      else if (!std::strcmp(s,"linear")) x->param_interp = (long)interp_mode::linear;
      else if (!std::strcmp(s,"lagrange4")) x->param_interp = (long)interp_mode::lagrange4;
      else if (!std::strcmp(s,"catmull") || !std::strcmp(s,"catmullrom") || !std::strcmp(s,"keys"))
        x->param_interp = (long)interp_mode::catmullrom;
      else if (!std::strcmp(s,"farrow3") || !std::strcmp(s,"farrow_cubic"))
        x->param_interp = (long)interp_mode::farrow3;
      else if (!std::strcmp(s,"farrow5") || !std::strcmp(s,"farrow_quintic"))
        x->param_interp = (long)interp_mode::farrow5;
    } else {
      long m = atom_getlong(av); if (m < 0) m = 0; if (m > 5) m = 5; x->param_interp = m;
    }
  }
  return MAX_ERR_NONE;
}
t_max_err primefir_attr_set_winshape(t_primefir* x, void*, long ac, t_atom* av) {
  if (ac && av) {
    if (atom_gettype(av) == A_SYM) {
      const char* s = atom_getsym(av)->s_name;
      if (!std::strcmp(s,"hann")) x->param_winshape = (long)winshape::hann;
      else if (!std::strcmp(s,"hamming")) x->param_winshape = (long)winshape::hamming;
      else if (!std::strcmp(s,"blackman")) x->param_winshape = (long)winshape::blackman;
      else if (!std::strcmp(s,"blackmanharris") || !std::strcmp(s,"bh4") || !std::strcmp(s,"bharris"))
        x->param_winshape = (long)winshape::blackmanharris;
      else if (!std::strcmp(s,"nuttall"))
        x->param_winshape = (long)winshape::nuttall;
      else if (!std::strcmp(s,"kaiser")) x->param_winshape = (long)winshape::kaiser;
    } else {
      long m = atom_getlong(av); if (m < 0) m = 0; if (m > 5) m = 5; x->param_winshape = m;
    }
    x->dirty = true;
  }
  return MAX_ERR_NONE;
}
t_max_err primefir_attr_set_kaiser_beta(t_primefir* x, void*, long ac, t_atom* av) {
  if (ac && av) { x->param_kaiser_beta = atom_getfloat(av); x->dirty = true; }
  return MAX_ERR_NONE;
}
t_max_err primefir_attr_set_keys_a(t_primefir* x, void*, long ac, t_atom* av) {
  if (ac && av) { x->param_keys_a = atom_getfloat(av); }
  return MAX_ERR_NONE;
}

// =====================  COMANDI  =====================
void primefir_clear(t_primefir* x)
{
  std::fill(std::begin(x->ringL), std::end(x->ringL), 0.0);
  std::fill(std::begin(x->ringR), std::end(x->ringR), 0.0);
  x->write_idx = 0;
}

// =====================  SEQUENZE  =====================
static inline double seq_value_d(const t_primefir* x, int n)
{
  if (n <= 0) return 0.0;
  switch (static_cast<seq_mode>(x->param_mode)) {
    case seq_mode::linear: return double(n);
    case seq_mode::prime:  return double(x->primes[std::min(n, x->primes_count)]);
    case seq_mode::phi:    return double(n) * kPhi;
    case seq_mode::sqrt2:  return double(n) * kSqrt2;
    case seq_mode::plastic:return double(n) * kPlastic;
    case seq_mode::pi:     return double(n) * M_PI;
    case seq_mode::e:      return double(n) * M_E;
    case seq_mode::prime_phi_index: {
      int idx = (int)std::floor(double(n) * kPhi);
      if (idx < 1) idx = 1;
      if (idx > x->primes_count) idx = x->primes_count;
      return double(x->primes[idx]);
    }
    case seq_mode::prime_phi_scaled:
      return double(x->primes[std::min(n, x->primes_count)]) * kPhi;
    default: return double(n);
  }
}

// =====================  FINESTRE RADIALI  =====================
// w(d), d=0..w-1 (picco a d=0, taper verso d=w-1)
static inline double window_value_radial(int d, int w, winshape ws, double kaiser_beta, double inv_i0beta)
{
  const int Dmax = w - 1;
  if (Dmax <= 0) return 1.0;

  const double th = M_PI * (double)d / (double)Dmax;  // 0..π
  switch (ws) {
    case winshape::hann: {
      // 0.5 * (1 + cos(π d/Dmax))
      return 0.5 * (1.0 + std::cos(th));
    }
    case winshape::hamming: {
      // 0.54 + 0.46 * cos(π d/Dmax)
      return 0.54 + 0.46 * std::cos(th);
    }
    case winshape::blackman: {
      // 0.42 + 0.5*cos(π d/Dmax) + 0.08*cos(2π d/Dmax)
      return 0.42 + 0.5 * std::cos(th) + 0.08 * std::cos(2.0 * th);
    }
    case winshape::blackmanharris: {
      // 4-term Blackman-Harris (radiale)
      const double a0 = 0.35875, a1 = 0.48829, a2 = 0.14128, a3 = 0.01168;
      return a0 + a1*std::cos(th) + a2*std::cos(2.0*th) + a3*std::cos(3.0*th);
    }
    case winshape::nuttall: {
      // 4-term Nuttall (radiale)
      const double a0 = 0.355768, a1 = 0.487396, a2 = 0.144232, a3 = 0.012604;
      return a0 + a1*std::cos(th) + a2*std::cos(2.0*th) + a3*std::cos(3.0*th);
    }
    case winshape::kaiser: {
      // I0(β * sqrt(1 - (d/Dmax)^2)) / I0(β)
      double r   = (double)d / (double)Dmax;
      double t   = std::sqrt(std::max(0.0, 1.0 - r * r));
      double num = i0_approx(kaiser_beta * t);
      return (inv_i0beta > 0.0 ? (num * inv_i0beta) : 1.0);
    }
    default: return 1.0;
  }
}

// =====================  KERNEL (DISTANZA-INDICIZZATO)  =====================
void primefir_update_kernel(t_primefir* x)
{
  const double overall = x->sr / 44100.0;

  // Mappatura "musicale": più risoluzione ai valori bassi
  double ww = std::pow(clamp01(x->param_window), 2.0);
  int w = (int)std::floor(2 + ww * (kMaxWindow - 2) * overall * 0.5);
  if (w < 2) w = 2;
  if (w > kMaxWindow) w = kMaxWindow;

  x->window = w;
  x->middle = (int)std::floor(std::sin(std::pow(clamp01(x->param_freq), 2.0) * (M_PI * 0.5)) * 0.5 * (double)w);

  // Frequenza relativa
  double f = std::pow(clamp01(x->param_freq), 2.0) * (M_PI * 0.5);
  if (f < 0.0001) f = 0.0001;
  const double f_rel = f / overall;

  // Primes necessari fino alla massima distanza
  int need_primes = (static_cast<seq_mode>(x->param_mode) == seq_mode::prime_phi_index)
                    ? std::min(kPrimeTableSize - 1, (int)std::ceil((w - 1) * kPhi) + 8)
                    : std::min(kPrimeTableSize - 1, (w - 1) + 8);
  if (!x->primes_ready || x->primes_count < need_primes)
    primefir_make_primes(x, need_primes);

  // Reset
  for (int i = 0; i < w; ++i) { x->fir[i] = 0.0; x->ioffs[i] = 0; x->ffrac[i] = 0.0; }

  const winshape ws = static_cast<winshape>(x->param_winshape);

  // d=0 (centro) non si somma
  x->fir[0]   = 0.0;
  x->ioffs[0] = 0;
  x->ffrac[0] = 0.0;

  const interp_mode imode = static_cast<interp_mode>(x->param_interp);

  // Costruzione sinc * finestra radiale su d=1..w-1
  double inv_i0beta = 0.0;
  if (ws == winshape::kaiser) {
    double den = i0_approx(x->param_kaiser_beta);
    inv_i0beta = (den > 0.0 ? (1.0 / den) : 0.0);
  }

  for (int d = 1; d < w; ++d) {
    const bool linear = (static_cast<seq_mode>(x->param_mode) == seq_mode::linear);
    const double off  = linear ? (double)d : seq_value_d(x, d);  // ritardo “tempo” a distanza d
    const double t    = off * f_rel;
    const double sinc = (t != 0.0) ? std::sin(t) / t : 1.0;

    const double win  = window_value_radial(d, w, ws, x->param_kaiser_beta, inv_i0beta);
    x->fir[d] = sinc * win;

    double D    = std::floor(off);
    double frac = off - D;
    if (D < 0.0) { D = 0.0; frac = 0.0; }
    if (D >= (double)(kRingSize - 6)) { D = (double)(kRingSize - 6); frac = 0.0; } // margine per farrow5
    if (imode == interp_mode::catmullrom && D < 1.0) {
      // Catmull–Rom usa il campione (wi - D + 1); garantiamo D>=1 così non si legge avanti.
      // Manteniamo l'offset totale spostando la frazione nella stessa storia disponibile.
      // Nota: se in futuro la sequenza generasse offset < 1, questo guardiano preserva la
      // precondizione sui campioni richiesti da ring_read_keys4.
      const double total = D + frac;
      D = 1.0;
      frac = total - 1.0;
      if (frac < 0.0) frac = 0.0;
      if (frac >= 1.0) frac = std::nextafter(1.0, 0.0);
    }
    x->ioffs[d] = (uint32_t)D;
    x->ffrac[d] = frac;
  }

  // Calibrazione (come tua logica originale)
  const double calib = (static_cast<seq_mode>(x->param_mode) == seq_mode::linear) ? 0.25 : 0.5;

  // Normalizzazione L1 dei tap (d>=1)
  double l1 = 0.0;
  for (int d = 1; d < w; ++d) l1 += std::abs(x->fir[d]);
  if (l1 < 1e-12) l1 = 1.0;

  const double norm  = x->param_normalize ? (1.0 / l1) : 1.0;
  const double gcomp = x->param_gaincomp ? std::sqrt(f_rel) : 1.0;

  x->post_scale = calib * norm * gcomp;
  x->dirty = false;
}

// =====================  DSP  =====================
void primefir_dsp64(t_primefir* x, t_object* dsp64, short*, double sr, long, long)
{
  x->sr = (sr > 0 ? sr : 44100.0);
  x->dirty = true;
  object_method(dsp64, gensym("dsp_add64"), x, (method)primefir_perform64, 0, NULL);
}

// ===== Interpolazioni di lettura dal ring =====
static inline double ring_read_linear(const double* ring, uint32_t wi, uint32_t D, double f)
{
  uint32_t i0 = (wi - D) & kRingMask;
  uint32_t i1 = (wi - D - 1) & kRingMask;
  return ring[i0] + f * (ring[i1] - ring[i0]);
}

// Lagrange 4‑tap causale: set {0,−1,−2,−3}
static inline double ring_read_lagrange4(const double* ring, uint32_t wi, uint32_t D, double mu)
{
  double mu2 = mu*mu, mu3 = mu2*mu;
  double w0 = -mu3*(1.0/6.0) + mu2 - (11.0/6.0)*mu + 1.0;
  double w1 =  mu3*0.5       - (5.0/2.0)*mu2 + 3.0*mu;
  double w2 = -mu3*0.5       + 2.0*mu2 - 1.5*mu;
  double w3 =  mu3*(1.0/6.0) - 0.5*mu2 + (1.0/3.0)*mu;

  uint32_t i0 = (wi - D)     & kRingMask;
  uint32_t i1 = (wi - D - 1) & kRingMask;
  uint32_t i2 = (wi - D - 2) & kRingMask;
  uint32_t i3 = (wi - D - 3) & kRingMask;

  return w0*ring[i0] + w1*ring[i1] + w2*ring[i2] + w3*ring[i3];
}

// Keys cubic (Catmull-Rom a=-0.5), set {−2,−1,0,+1}
static inline double keys_h(double a, double x)
{
  x = std::abs(x);
  if (x < 1.0) {
    return ((a+2.0)*x - (a+3.0))*x*x + 1.0;
  } else if (x < 2.0) {
    return (((a*x - 5.0*a)*x + 8.0*a)*x - 4.0*a);
  } else {
    return 0.0;
  }
}
static inline double ring_read_keys4(const double* ring, uint32_t wi, uint32_t D, double mu, double a)
{
  double w_m2 = keys_h(a, 2.0 - mu);
  double w_m1 = keys_h(a, 1.0 - mu);
  double w_0  = keys_h(a, mu);
  double w_p1 = keys_h(a, 1.0 + mu);

  uint32_t im2 = (wi - D - 2) & kRingMask;
  uint32_t im1 = (wi - D - 1) & kRingMask;
  uint32_t i0  = (wi - D)     & kRingMask;
  uint32_t ip1 = (wi - D + 1) & kRingMask; // ≤ wi perché D>=1

  return w_m2*ring[im2] + w_m1*ring[im1] + w_0*ring[i0] + w_p1*ring[ip1];
}

// ===== Farrow polinomiale generico (Lagrange-form) =====
// Ordine P: usa i campioni {0, -1, -2, ..., -P}; valuta f(μ), μ∈[0,1)
// w_k(μ) = prod_{m≠k} (μ - m) / prod_{m≠k} (k - m)  con k=0..P
static inline void farrow_weights_lagrange(int P, double mu, double* w /*size P+1*/)
{
  // denom_k = (-1)^{P-k} k! (P-k)!
  static const double fact[] = {1,1,2,6,24,120,720};
  for (int k = 0; k <= P; ++k) {
    double num = 1.0;
    for (int m = 0; m <= P; ++m) if (m != k) num *= (mu - (double)m);
    double denom = fact[k] * fact[P - k];
    if ( ((P - k) & 1) ) denom = -denom; // (-1)^{P-k}
    w[k] = (denom != 0.0) ? (num / denom) : 0.0;
  }
}
static inline double ring_read_farrow_generic(const double* ring, uint32_t wi, uint32_t D, double mu, int P)
{
  double w[6]; // supporta fino a P=5
  farrow_weights_lagrange(P, mu, w);
  double acc = 0.0;
  for (int k = 0; k <= P; ++k) {
    uint32_t idx = (wi - D - (uint32_t)k) & kRingMask;
    acc += w[k] * ring[idx];
  }
  return acc;
}

void primefir_perform64(t_primefir* x, t_object*, double** ins, long numins,
                        double** outs, long numouts, long sampleframes, long, void*)
{
  if (x->ob.z_disabled) return;

  double* inL  = (numins >= 1 && ins[0]) ? ins[0] : nullptr;
  double* inR  = (numins >= 2 && ins[1]) ? ins[1] : nullptr;
  double* outL = (numouts >= 1 && outs[0]) ? outs[0] : nullptr;
  double* outR = (numouts >= 2 && outs[1]) ? outs[1] : nullptr;
  if (!inL || !outL || !outR) return;
  if (!inR) inR = inL;

  if (x->dirty) primefir_update_kernel(x);

  const int w = x->window;
  const double scale = x->post_scale;
  const interp_mode imode = static_cast<interp_mode>(x->param_interp);
  const double keys_a = x->param_keys_a;

  uint32_t wi = x->write_idx;

  for (long n = 0; n < sampleframes; ++n) {
    double sL = inL[n], sR = inR[n];
    if (std::abs(sL) < kTiny) sL = 0.0;
    if (std::abs(sR) < kTiny) sR = 0.0;

    x->ringL[wi] = sL;
    x->ringR[wi] = sR;

    double accL = 0.0, accR = 0.0;

    for (int d = 1; d < w; ++d) {
      const uint32_t D = x->ioffs[d];
      const double   f = x->ffrac[d];

      double vL, vR;
      switch (imode) {
        case interp_mode::linear:
          vL = ring_read_linear(x->ringL, wi, D, f);
          vR = ring_read_linear(x->ringR, wi, D, f);
          break;
        case interp_mode::lagrange4:
          vL = ring_read_lagrange4(x->ringL, wi, D, f);
          vR = ring_read_lagrange4(x->ringR, wi, D, f);
          break;
        case interp_mode::catmullrom:
          vL = ring_read_keys4(x->ringL, wi, D, f, keys_a);
          vR = ring_read_keys4(x->ringR, wi, D, f, keys_a);
          break;
        case interp_mode::farrow3:
          vL = ring_read_farrow_generic(x->ringL, wi, D, f, 3); // P=3 (4 tap)
          vR = ring_read_farrow_generic(x->ringR, wi, D, f, 3);
          break;
        case interp_mode::farrow5:
          vL = ring_read_farrow_generic(x->ringL, wi, D, f, 5); // P=5 (6 tap)
          vR = ring_read_farrow_generic(x->ringR, wi, D, f, 5);
          break;
        default: { // off
          uint32_t idx = (wi - D) & kRingMask;
          vL = x->ringL[idx];
          vR = x->ringR[idx];
        } break;
      }

      // robustezza numerica (rari casi estremi)
      if (!std::isfinite(vL)) vL = 0.0;
      if (!std::isfinite(vR)) vR = 0.0;

      const double c = x->fir[d];
      accL += vL * c;
      accR += vR * c;
    }

    outL[n] = accL * scale;
    outR[n] = accR * scale;

    wi = (wi + 1u) & kRingMask;
  }
  x->write_idx = wi;
}

// =====================  PRIMI (sieve)  =====================
static int prime_upper_bound_from_count(int count_needed)
{
  if (count_needed <= 0) return 2;

  static const int kSmallPrimeBounds[] = {2, 3, 5, 7, 11, 13};
  if (count_needed <= 6) return kSmallPrimeBounds[count_needed - 1];

  const double dn      = static_cast<double>(count_needed);
  const double logn    = std::log(dn);
  const double loglogn = std::log(logn);

  // Dusart upper bound n (log n + log log n) for n >= 6, with an extra margin
  // (10% + constant) to guard against rounding and small deviations.
  const double estimate = dn * (logn + loglogn);
  const double margin   = std::max(16.0, estimate * 0.1);

  const int limit = static_cast<int>(std::ceil(estimate + margin));
  return std::max(limit, kSmallPrimeBounds[5]);
}

void primefir_make_primes(t_primefir* x, int count_needed)
{
  if (x->primes_ready && x->primes_count >= count_needed) return;

  const int capped_count = std::max(1, std::min(count_needed, kPrimeTableSize - 1));
  const int limit = prime_upper_bound_from_count(capped_count);
  std::vector<bool> is_composite(limit + 1, false);
  std::vector<int>  plist; plist.reserve(std::min(count_needed + 64, kPrimeTableSize));

  for (int p = 2; p * p <= limit; ++p)
    if (!is_composite[p]) for (int q = p * p; q <= limit; q += p) is_composite[q] = true;
  for (int n = 2; n <= limit && (int)plist.size() < count_needed; ++n)
    if (!is_composite[n]) plist.push_back(n);

  x->primes[0] = 0;
  x->primes_count = std::min((int)plist.size(), kPrimeTableSize - 1);
  for (int i = 1; i <= x->primes_count; ++i) x->primes[i] = plist[i-1];
  x->primes_ready = true;
}
