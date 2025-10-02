import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import json
import glob

def create_2d_sideband_comparison():
    """Create 2D overlay of all message spectra with carrier and sidebands marked"""
    
    # Load all result files
    json_files = glob.glob('nanogpt_*_data.json')
    
    if len(json_files) == 0:
        print("No data files found!")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    all_peaks = []
    
    for idx, json_file in enumerate(sorted(json_files)):
        with open(json_file, 'r') as f:
            result = json.load(f)
        
        message = result['message']
        
        # Compute FFT
        signal = np.array(result['actual_diversities'])
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        fft_result = fft(signal)
        freqs = fftfreq(len(signal), d=1.0)
        
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_magnitudes = np.abs(fft_result[pos_mask])
        
        # Find peak
        peak_idx = np.argmax(pos_magnitudes)
        detected_freq = pos_freqs[peak_idx]
        all_peaks.append(detected_freq)
        
        # Plot spectrum
        ax.plot(pos_freqs, pos_magnitudes, linewidth=2, 
               label=f'{message} (peak: {detected_freq:.3f})', 
               color=colors[idx % len(colors)], alpha=0.8)
        
        print(f"{message}: Detected frequency = {detected_freq:.3f}")
    
    # Mark expected carrier
    expected_carrier = 1.0 / 3.0
    ax.axvline(x=expected_carrier, color='red', linestyle='--', 
              linewidth=2.5, label=f'Expected Carrier: {expected_carrier:.3f}', zorder=10)
    
    # Mark average detected carrier
    avg_detected = np.mean(all_peaks)
    ax.axvline(x=avg_detected, color='darkred', linestyle=':', 
              linewidth=2, label=f'Avg Detected: {avg_detected:.3f}', alpha=0.7)
    
    # Estimate and mark sidebands (if visible)
    # Look for secondary peaks around carrier
    carrier_region = (pos_freqs > expected_carrier - 0.15) & (pos_freqs < expected_carrier + 0.15)
    
    # Find local maxima in carrier region
    from scipy.signal import find_peaks
    for idx, json_file in enumerate(sorted(json_files)):
        with open(json_file, 'r') as f:
            result = json.load(f)
        signal = np.array(result['actual_diversities'])
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        fft_result = fft(signal)
        freqs = fftfreq(len(signal), d=1.0)
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_magnitudes = np.abs(fft_result[pos_mask])
        
        peaks, _ = find_peaks(pos_magnitudes, height=np.max(pos_magnitudes) * 0.3, distance=5)
        peak_freqs = pos_freqs[peaks]
        
        # Find sidebands near carrier
        lower_sb = peak_freqs[peak_freqs < expected_carrier]
        upper_sb = peak_freqs[peak_freqs > expected_carrier]
        
        if len(lower_sb) > 0:
            closest_lower = lower_sb[np.argmax(lower_sb)]
            if idx == 0:  # Only mark once
                ax.axvline(x=closest_lower, color='purple', linestyle=':', 
                          linewidth=1.5, label=f'Lower SB: {closest_lower:.3f}', alpha=0.6)
        
        if len(upper_sb) > 0:
            closest_upper = upper_sb[np.argmin(upper_sb)]
            if idx == 0:  # Only mark once
                ax.axvline(x=closest_upper, color='orange', linestyle=':', 
                          linewidth=1.5, label=f'Upper SB: {closest_upper:.3f}', alpha=0.6)
    
    ax.set_xlabel('Frequency (cycles/step)', fontsize=13)
    ax.set_ylabel('Magnitude', fontsize=13)
    ax.set_title('Sideband Structure (2D): Lexical Diversity Modulation', 
                fontsize=15, fontweight='bold')
    ax.set_xlim(0.15, 0.50)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('lexical_diversity_sideband_2d.png', dpi=300, bbox_inches='tight')
    print("\nSaved: lexical_diversity_sideband_2d.png")
    plt.show()

if __name__ == "__main__":
    create_2d_sideband_comparison()
