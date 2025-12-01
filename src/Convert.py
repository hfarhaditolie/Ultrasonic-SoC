import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import pandas as pd
import os

# Configuration
actuator = 1
reciever = 2
nfft = 256
window = get_window('hann', nfft)
noverlap = 0

# Create output directories
output_dirs = {
    'waveform': '../data/waveform_images',
}

for dir_name in output_dirs.values():
    os.makedirs(dir_name, exist_ok=True)

# Load and prepare the dataset in bi-directional direction
file_path1 = f'../data/charging/signal3_4_SoC_raw.csv'
file_path2 = f'../data/discharging/signal4_3_SoCD_raw.csv'

# Read both CSV files into DataFrames
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Convert combined DataFrame to numpy array
data1 = np.asarray(df1)
data2 = np.asarray(df2)

y1 = data1[:, -1]  # SoC at charging state
X1 = data1[:, :-1]

y2 = data2[:, -1]  # SoC at discharging state
X2 = data2[:, :-1]

signal_data = np.vstack((X1, X2))
SoCs = np.hstack((y1, y2))
np.save("SoCs.npy",SoCs) #SoCs for all signals
sample_rate = 100e6
def save_figure(fig, dir_name, idx):
    """Helper function to save figures"""
    filename = os.path.join(output_dirs[dir_name], f"{dir_name}_{idx:03d}.png")
    fig.savefig(filename, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

for idx, signal in enumerate(signal_data):
    # 1. Waveform Visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.linspace(0, len(signal)/sample_rate, len(signal)), signal)
    ax.axis('off')
    save_figure(fig, 'waveform', idx)
    
