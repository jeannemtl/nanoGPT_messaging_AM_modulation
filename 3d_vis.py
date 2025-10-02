import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.fft import fft, fftfreq
import json
import glob

def create_3d_lexical_diversity_gif():
    """Create 3D rotating visualization of lexical diversity spectra"""
    
    # Load all result files
    json_files = glob.glob('nanogpt_*_data.json')
    
    if len(json_files) < 2:
        print("Need at least 2 message files for 3D visualization")
        print("Run generator.py with different messages first")
        return
    
    print(f"Found {len(json_files)} result files")
    
    messages = []
    all_spectra = []
    
    for json_file in sorted(json_files):
        with open(json_file, 'r') as f:
            result = json.load(f)
        
        message = result['message']
        messages.append(message)
        
        # Compute FFT
        signal = np.array(result['actual_diversities'])
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        fft_result = fft(signal)
        freqs = fftfreq(len(signal), d=1.0)
        
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_magnitudes = np.abs(fft_result[pos_mask])
        
        all_spectra.append((pos_freqs, pos_magnitudes))
        print(f"  Loaded: {message}")
    
    # Create 3D mesh
    X = all_spectra[0][0]  # Frequencies
    Y = np.arange(len(messages))  # Message index
    Z = np.array([spectrum[1] for spectrum in all_spectra])
    
    X_grid, Y_grid = np.meshgrid(X, Y)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(frame):
        ax.clear()
        
        # Plot surface
        surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', 
                              alpha=0.8, edgecolor='none')
        
        # Mark carrier frequency plane
        expected_carrier = 1.0 / 3.0
        carrier_x = np.ones_like(Z) * expected_carrier
        ax.plot_surface(carrier_x, Y_grid, Z * 0.3, color='red', alpha=0.3)
        
        # Plot individual traces
        for idx, (freqs, mags) in enumerate(all_spectra):
            ax.plot(freqs, [idx]*len(freqs), mags, 'k-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Frequency (cycles/step)', fontsize=11)
        ax.set_ylabel('Message', fontsize=11)
        ax.set_zlabel('Magnitude', fontsize=11)
        ax.set_title('3D Frequency Spectrum: Lexical Diversity', fontsize=13, fontweight='bold')
        ax.set_yticks(Y)
        ax.set_yticklabels(messages)
        ax.set_xlim(0, 0.5)
        
        # Rotate view
        ax.view_init(elev=25, azim=frame)
        
        return surf,
    
    print("\nCreating animation...")
    anim = FuncAnimation(fig, animate, frames=np.arange(0, 360, 2), 
                        interval=50, blit=False)
    
    writer = PillowWriter(fps=20)
    anim.save('lexical_diversity_3d_rotation.gif', writer=writer, dpi=100)
    print("Saved: lexical_diversity_3d_rotation.gif")
    plt.close()

if __name__ == "__main__":
    create_3d_lexical_diversity_gif()
