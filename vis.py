import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import json
import sys

def visualize_results(result, save_prefix='nanogpt'):
    """Visualize lexical diversity modulation and FFT analysis"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    steps = np.arange(len(result['actual_diversities']))
    
    # 1. Diversity over time
    axes[0].plot(steps, result['target_diversities'], 'r--', 
                linewidth=2, label='Target', alpha=0.7)
    axes[0].plot(steps, result['actual_diversities'], 'g-', 
                linewidth=2, label='Actual')
    axes[0].set_ylabel('Lexical Diversity')
    axes[0].set_title(f'Lexical Diversity Modulation (Message: {result["message"]})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. FFT Spectrum
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
    expected_freq = 1.0 / 3.0  # carrier_freq = 3
    
    axes[1].plot(pos_freqs, pos_magnitudes, 'b-', linewidth=2)
    axes[1].axvline(x=expected_freq, color='r', linestyle='--', 
                   linewidth=2, label=f'Expected: {expected_freq:.3f}')
    axes[1].axvline(x=detected_freq, color='g', linestyle=':', 
                   linewidth=2, label=f'Detected: {detected_freq:.3f}')
    axes[1].set_xlabel('Frequency (cycles/step)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('FFT Spectrum')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 0.5)
    
    # 3. Sample text snippets
    axes[2].axis('off')
    sample_indices = [0, len(steps)//2, len(steps)-1]
    text_info = "Sample Generated Text:\n\n"
    for i in sample_indices:
        div = result['actual_diversities'][i]
        text = result['reasoning_steps'][i][:60]
        text_info += f"Step {i} (div={div:.3f}):\n{text}...\n\n"
    
    axes[2].text(0.1, 0.5, text_info, fontsize=10, 
                verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_prefix}_analysis.png")
    plt.show()
    
    # Save results to JSON
    output = {
        'message': result['message'],
        'correlation': float(result['correlation']),
        'detected_frequency': float(detected_freq),
        'expected_frequency': float(expected_freq),
        'frequency_error': float(abs(detected_freq - expected_freq)),
        'reasoning_steps': result['reasoning_steps'],
        'actual_diversities': [float(x) for x in result['actual_diversities']]
    }
    
    with open(f'{save_prefix}_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {save_prefix}_results.json")
    
    return output

def load_and_visualize(json_file):
    """Load result from JSON and visualize"""
    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        result = json.load(f)
    
    # Extract prefix from filename
    prefix = json_file.replace('.json', '').replace('_data', '')
    
    visualize_results(result, save_prefix=prefix)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Visualize from JSON file
        load_and_visualize(sys.argv[1])
    else:
        print("Usage: python3 visualize_nanogpt.py <result_file.json>")
        print("Or import and use visualize_results(result) in your generator script")
