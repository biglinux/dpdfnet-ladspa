//! High-frequency reconstruction for the split-band variants.
//!
//! The 16 kHz DPDFNet networks only see 0–8 kHz. Run inside a 48 kHz graph
//! through this wrapper, the STFT is 48 kHz (nfft 960) and the network is fed
//! the low 161 bins; everything above 8 kHz (bins 161..481) never reaches the
//! model. This processor fills that band so the output is full-band instead of
//! telephone-band.
//!
//! Two modes, chosen per frame by the high band's own SNR:
//!
//! - **Spectral gate** — when the original >8 kHz content is clean enough to
//!   keep, it is passed through, gated by the speech probability and a per-bin
//!   SNR gate, with a short bypass after a transient so consonants ("t", "s",
//!   "k") keep their edge.
//! - **Air exciter** — when the high band is too noisy to keep, it is discarded
//!   and synthesised from the enhanced 4–8 kHz band (second harmonic plus a
//!   −6 dB/octave tilt), renormalised against the clean 6–8 kHz reference.
//!
//! Ported from the GTCRN LADSPA wrapper (BigLinux, MIT), which proved the
//! approach; the bin numbers are recomputed for DPDFNet's 50 Hz/bin geometry
//! (GTCRN runs 31.25 Hz/bin).

use realfft::num_complex::Complex;

/// Reconstructs the band the model does not see.
pub struct HighBand {
    /// Number of low (model) bins — the boundary the high band starts after.
    low_bins: usize,
    /// Hz per FFT bin, for turning frequencies into bin indices.
    bin_hz: f32,
    /// Previous-frame high-band magnitudes, for spectral-flux transient
    /// detection.
    prev_mag: Vec<f32>,
    has_prev: bool,
    /// Frames of gate bypass left after a transient, and the full hold length.
    transient_hold: usize,
    transient_hold_frames: usize,
    /// Adaptive per-bin noise floor, learned during silence.
    noise_floor: Vec<f32>,
    noise_initialised: bool,
}

impl HighBand {
    /// `hf_bins` is the number of bins above the model cutoff (STFT bins minus
    /// model bins); `bin_hz` is `host_rate / stft_win_len` (the FFT resolution).
    pub fn new(low_bins: usize, hf_bins: usize, hop: usize, host_rate: usize, bin_hz: f32) -> Self {
        let frame_dur_s = hop as f64 / host_rate as f64;
        let hold_frames = (0.020 / frame_dur_s).ceil() as usize;
        Self {
            low_bins,
            bin_hz,
            prev_mag: vec![0.0; hf_bins],
            has_prev: false,
            transient_hold: 0,
            transient_hold_frames: hold_frames.max(1),
            noise_floor: vec![1e-6; hf_bins],
            noise_initialised: false,
        }
    }

    /// Fill `output` (the high band) from the original high band and the clean
    /// low band. `speech` is 0.0 (silence) .. 1.0 (speech).
    ///
    /// `hf_original` and `output` are the STFT bins above the model cutoff;
    /// `enhanced_low` is the model's clean low band, complex, `low_bins` long.
    pub fn process(
        &mut self,
        hf_original: &[Complex<f32>],
        enhanced_low: &[Complex<f32>],
        speech: f32,
        output: &mut [Complex<f32>],
    ) {
        self.update_noise_floor(hf_original, speech);

        if self.detect_transient(hf_original) {
            self.transient_hold = self.transient_hold_frames;
        } else if self.transient_hold > 0 {
            self.transient_hold -= 1;
        }

        if self.high_band_snr(hf_original) < 2.0 && self.noise_initialised {
            self.synthesize_air(enhanced_low, output);
        } else {
            self.spectral_gate(hf_original, speech, output);
        }
    }

    /// Half-wave-rectified spectral flux: a transient spikes positive flux
    /// across many bins at once.
    fn detect_transient(&mut self, hf: &[Complex<f32>]) -> bool {
        let count = hf.len().min(self.prev_mag.len());
        if count == 0 {
            return false;
        }
        let mut flux = 0.0_f32;
        let mut avg_mag = 0.0_f32;
        for (h, prev) in hf[..count].iter().zip(self.prev_mag[..count].iter_mut()) {
            let mag = h.norm();
            avg_mag += mag;
            let diff = mag - *prev;
            if diff > 0.0 {
                flux += diff;
            }
            *prev = mag;
        }
        avg_mag /= count as f32;
        let was_valid = self.has_prev;
        self.has_prev = true;
        if !was_valid || avg_mag < 1e-10 {
            return false;
        }
        flux / (avg_mag * count as f32) > 0.35
    }

    /// Pass the original high band through, gated by speech probability and a
    /// per-bin SNR gate; bypass the gate briefly after a transient.
    fn spectral_gate(&self, hf: &[Complex<f32>], speech: f32, output: &mut [Complex<f32>]) {
        let bypassed = self.transient_hold > 0;
        let bypass_floor = if bypassed { 0.5 } else { 0.0 };
        let envelope = if bypassed || speech > 0.7 {
            1.0
        } else if speech > 0.1 {
            (speech - 0.1) / 0.6
        } else {
            0.0
        };
        let count = hf.len().min(output.len());
        let snr_threshold = 1.0;
        let snr_range = 3.5;

        if self.noise_initialised {
            for i in 0..count {
                let snr = hf[i].norm() / (self.noise_floor[i] + 1e-10);
                let snr_gain = ((snr - snr_threshold) / snr_range).clamp(0.0, 1.0);
                let gain = if bypassed {
                    snr_gain.max(bypass_floor)
                } else {
                    envelope * snr_gain
                };
                output[i] = Complex::new(hf[i].re * gain, hf[i].im * gain);
            }
        } else {
            let atten = envelope * 0.5;
            for i in 0..count {
                output[i] = Complex::new(hf[i].re * atten, hf[i].im * atten);
            }
        }
        for o in output.iter_mut().skip(count) {
            *o = Complex::new(0.0, 0.0);
        }
    }

    /// Synthesise the high band from the enhanced 4–8 kHz content when the
    /// original is too noisy to keep. Mirrors that band up, generates a second
    /// harmonic, tilts −6 dB/octave, renormalises against the clean 6–8 kHz
    /// reference.
    fn synthesize_air(&self, enhanced_low: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let hf_count = output.len();
        output.fill(Complex::new(0.0, 0.0));

        // Source is 4–8 kHz of the clean low band; reference is 6–8 kHz.
        let src_start = ((4000.0 / self.bin_hz).round() as usize).min(self.low_bins);
        let ref_start = ((6000.0 / self.bin_hz).round() as usize).min(self.low_bins);
        let src_len = self.low_bins.saturating_sub(src_start).max(1);

        let ref_energy: f32 = enhanced_low[ref_start..self.low_bins]
            .iter()
            .map(|c| c.norm())
            .sum::<f32>()
            / (self.low_bins - ref_start).max(1) as f32;
        if ref_energy < 1e-10 {
            return;
        }

        for (i, out) in output.iter_mut().enumerate().take(hf_count) {
            let src = enhanced_low[src_start + (i % src_len)];
            let mag = src.norm();
            let harmonic = mag * mag; // second harmonic
            let freq = 8000.0 + i as f32 * self.bin_hz;
            let tilt = 1.0 / (freq / 8000.0); // −6 dB/octave above 8 kHz
            let phase = src.im.atan2(src.re);
            let final_mag = harmonic * tilt;
            *out = Complex::new(final_mag * phase.cos(), final_mag * phase.sin());
        }

        let synth_energy: f32 =
            output.iter().map(|c| c.norm()).sum::<f32>() / hf_count.max(1) as f32;
        if synth_energy > 1e-10 {
            let ratio = (ref_energy * 0.4 / synth_energy).min(3.0);
            for c in output.iter_mut() {
                c.re *= ratio;
                c.im *= ratio;
            }
        }
    }

    /// Learn the per-bin high-band noise floor while there is no speech.
    fn update_noise_floor(&mut self, hf: &[Complex<f32>], speech: f32) {
        if speech >= 0.1 {
            return;
        }
        let count = hf.len().min(self.noise_floor.len());
        for (h, floor) in hf[..count].iter().zip(self.noise_floor[..count].iter_mut()) {
            let mag = h.norm();
            *floor = if self.noise_initialised {
                0.99 * *floor + 0.01 * mag
            } else {
                mag
            };
        }
        self.noise_initialised = true;
    }

    /// Mean per-bin SNR of the high band against the learned floor.
    fn high_band_snr(&self, hf: &[Complex<f32>]) -> f32 {
        if !self.noise_initialised {
            return 10.0;
        }
        let count = hf.len().min(self.noise_floor.len());
        if count == 0 {
            return 0.0;
        }
        let total: f32 = hf[..count]
            .iter()
            .enumerate()
            .map(|(i, c)| c.norm() / (self.noise_floor[i] + 1e-10))
            .sum();
        total / count as f32
    }
}

/// Smoothed speech probability from the low band's input vs. enhanced energy.
///
/// The high-band processor needs a 0..1 speech gate; the DPDFNet network does
/// not emit one, so derive it the way the GTCRN wrapper does — from energy.
/// The denoiser keeps speech and removes noise, so the enhanced-to-input energy
/// ratio is high during voice and low in a pause; an onset over a tracked input
/// floor primes the gate so the first syllable is not gated out.
pub struct SpeechGate {
    input_floor: f32,
    gate: f32,
}

impl SpeechGate {
    pub fn new() -> Self {
        Self {
            input_floor: 1e-6,
            gate: 0.0,
        }
    }

    /// One hop. `input_energy` and `enhanced_energy` are sums of squared
    /// magnitude over the low band. Returns the 0..1 gate.
    pub fn update(&mut self, input_energy: f32, enhanced_energy: f32) -> f32 {
        let ratio = enhanced_energy / (input_energy + 1e-9);
        let onset = input_energy > self.input_floor * 2.5;
        let speaking = onset && ratio > 0.05;
        if speaking {
            self.gate = 1.0;
        } else {
            self.gate *= 0.95;
            if self.gate < 0.01 {
                self.gate = 0.0;
            }
        }
        // Track the input floor only when confidently silent.
        if self.gate < 0.1 {
            self.input_floor = 0.95 * self.input_floor + 0.05 * input_energy;
        }
        self.gate
    }
}
