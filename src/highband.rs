//! High-band handling for the split-band variants.
//!
//! The 16 kHz DPDFNet networks only see 0–8 kHz. Run inside a 48 kHz graph
//! through this wrapper, the STFT is 48 kHz (nfft 960) and the network is fed
//! the low 161 bins; everything above 8 kHz (bins 161..481) never reaches the
//! model. This passes that captured band through, gated by the speech
//! probability and a per-bin SNR gate so background noise above 8 kHz is
//! attenuated while the real high band — fricatives, air — is kept. A short
//! bypass after a transient keeps consonants ("t", "s", "k") from being gated.
//!
//! It deliberately does **not** synthesize the high band. An earlier port
//! carried the GTCRN air exciter, which mirrors 4–8 kHz up and squares it for a
//! second harmonic when the high band's SNR is low. Measured on speech, that
//! fired on 68–89 % of frames — even on clean speech, because voice carries
//! almost no energy above 8 kHz outside fricatives, so the SNR is low most of
//! the time. That meant the "full-band" output was mostly fabricated, not the
//! captured voice, which is the opposite of what a natural-capture denoiser
//! should do. A synthetic highs effect, if ever wanted, belongs in its own
//! opt-in stage, not here.
//!
//! Ported from the GTCRN LADSPA wrapper (BigLinux, MIT).

use realfft::num_complex::Complex;

/// Passes and gates the band the model does not see.
pub struct HighBand {
    /// Previous-frame high-band magnitudes, for spectral-flux transient
    /// detection.
    prev_mag: Vec<f32>,
    has_prev: bool,
    /// Frames of gate bypass left after a transient, and the full hold length.
    transient_hold: usize,
    transient_hold_frames: usize,
    /// Adaptive per-bin noise floor, learned during silence, for the SNR gate.
    noise_floor: Vec<f32>,
    noise_initialised: bool,
}

impl HighBand {
    /// `hf_bins` is the number of bins above the model cutoff (STFT bins minus
    /// model bins).
    pub fn new(hf_bins: usize, hop: usize, host_rate: usize) -> Self {
        let frame_dur_s = hop as f64 / host_rate as f64;
        let hold_frames = (0.020 / frame_dur_s).ceil() as usize;
        Self {
            prev_mag: vec![0.0; hf_bins],
            has_prev: false,
            transient_hold: 0,
            transient_hold_frames: hold_frames.max(1),
            noise_floor: vec![1e-6; hf_bins],
            noise_initialised: false,
        }
    }

    /// Gate the captured high band into `output`. `speech` is 0.0 (silence) ..
    /// 1.0 (speech). `hf_original` and `output` are the STFT bins above the
    /// model cutoff.
    pub fn process(
        &mut self,
        hf_original: &[Complex<f32>],
        speech: f32,
        output: &mut [Complex<f32>],
    ) {
        self.update_noise_floor(hf_original, speech);

        if self.detect_transient(hf_original) {
            self.transient_hold = self.transient_hold_frames;
        } else if self.transient_hold > 0 {
            self.transient_hold -= 1;
        }

        self.spectral_gate(hf_original, speech, output);
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
}

/// Smoothed speech probability from the low band's input vs. enhanced energy.
///
/// The high-band gate needs a 0..1 speech gate; the DPDFNet network does not
/// emit one, so derive it the way the GTCRN wrapper does — from energy. The
/// denoiser keeps speech and removes noise, so the enhanced-to-input energy
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
