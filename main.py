import torch
import torch.nn as nn
import serial
import numpy as np
import time
from scipy import signal
import matplotlib.pyplot as plt

# ===== MODEL DEFINITION =====
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=32, stride=1, padding='same'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class ApneaCNN(nn.Module):
    def __init__(self, input_channels=1, input_length=12000):
        super().__init__()
        self.conv_layers = nn.Sequential(
            ConvBlock(input_channels, 45, kernel_size=32),
            ConvBlock(45, 45, kernel_size=32),
            ConvBlock(45, 45, kernel_size=32),
            ConvBlock(45, 45, kernel_size=32),
            ConvBlock(45, 45, kernel_size=32),
            ConvBlock(45, 45, kernel_size=32),
            ConvBlock(45, 45, kernel_size=32),
            ConvBlock(45, 45, kernel_size=32),
            ConvBlock(45, 45, kernel_size=32),
            ConvBlock(45, 45, kernel_size=32),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_length)
            dummy_out = self.conv_layers(dummy)
            self.flattened_size = dummy_out.numel()

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flattened_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


# ===== PREPROCESSING FUNCTIONS =====
def preprocess_ecg(raw_data, target_length=12000, target_fs=100):
    """
    Preprocess raw ECG data from ESP32
    
    Args:
        raw_data: numpy array of raw ADC values (0-4095)
        target_length: desired length of signal (12000 samples = 120 seconds at 100Hz)
        target_fs: target sampling frequency
    
    Returns:
        preprocessed numpy array ready for model
    """
    # Convert ADC values to voltage (0-4095 → 0-3.3V)
    ecg_signal = (raw_data / 4095.0) * 3.3
    
    # Normalize to zero mean and unit variance
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)
    
    # Apply bandpass filter (0.5 - 40 Hz) to remove noise
    sos = signal.butter(4, [0.5, 40], btype='band', fs=target_fs, output='sos')
    ecg_signal = signal.sosfilt(sos, ecg_signal)
    
    # Resample or pad to target length
    current_length = len(ecg_signal)
    if current_length < target_length:
        # Pad with zeros
        ecg_signal = np.pad(ecg_signal, (0, target_length - current_length), mode='constant')
    elif current_length > target_length:
        # Truncate
        ecg_signal = ecg_signal[:target_length]
    
    return ecg_signal


# ===== ESP32 COMMUNICATION =====
class ECGDataCollector:
    def __init__(self, port, baudrate=115200):
        """
        Initialize serial connection to ESP32
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Baud rate (default 115200)
        """
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for connection to establish
        print(f"✅ Connected to {port}")
        
    def collect_ecg_segment(self, num_samples=1200, timeout=30):
        """
        Collect ECG data from ESP32
        
        Args:
            num_samples: Number of samples to collect
            timeout: Maximum time to wait (seconds)
        
        Returns:
            numpy array of ECG values
        """
        print("📡 Requesting ECG data from ESP32...")
        self.ser.write(b'START\n')
        
        ecg_data = []
        start_time = time.time()
        recording = False
        
        while len(ecg_data) < num_samples:
            if time.time() - start_time > timeout:
                raise TimeoutError("Data collection timeout")
            
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                
                if line == "RECORDING_START":
                    recording = True
                    print("🔴 Recording started...")
                    continue
                    
                if line == "RECORDING_END":
                    print("✅ Recording complete!")
                    break
                    
                if line.startswith("ERROR:"):
                    raise Exception(f"ESP32 Error: {line}")
                
                if recording:
                    try:
                        value = int(line)
                        ecg_data.append(value)
                        if len(ecg_data) % 100 == 0:
                            print(f"  Collected {len(ecg_data)}/{num_samples} samples")
                    except ValueError:
                        continue
        
        return np.array(ecg_data)
    
    def close(self):
        self.ser.close()
        print("🔌 Serial connection closed")


# ===== MAIN INFERENCE PIPELINE =====
def run_ecg_inference(model_path, serial_port, device='cpu'):
    """
    Complete pipeline: Collect ECG → Preprocess → Inference
    
    Args:
        model_path: Path to saved model checkpoint
        serial_port: Serial port for ESP32
        device: 'cpu' or 'cuda'
    """
    # Load model
    print("🔧 Loading model...")
    model = ApneaCNN(input_channels=1, input_length=12000)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!\n")
    
    # Connect to ESP32
    collector = ECGDataCollector(serial_port)
    
    try:
        while True:
            print("\n" + "="*60)
            print("Press ENTER to start ECG recording (or 'q' to quit)")
            user_input = input()
            if user_input.lower() == 'q':
                break
            
            # Collect ECG data
            raw_ecg = collector.collect_ecg_segment(num_samples=1200)
            
            # Preprocess
            print("\n🔄 Preprocessing ECG signal...")
            processed_ecg = preprocess_ecg(raw_ecg, target_length=12000)
            
            # Convert to tensor
            ecg_tensor = torch.FloatTensor(processed_ecg).unsqueeze(0).unsqueeze(0)  # (1, 1, 12000)
            ecg_tensor = ecg_tensor.to(device)
            
            # Inference
            print("🧠 Running inference...")
            with torch.no_grad():
                prediction = model(ecg_tensor)
                prob = prediction.item()
            
            # Display results
            print("\n" + "="*60)
            print("📊 RESULTS:")
            print(f"   Apnea Probability: {prob:.4f} ({prob*100:.2f}%)")
            if prob > 0.5:
                print("   ⚠️  APNEA DETECTED")
            else:
                print("   ✅ Normal ECG")
            print("="*60)
            
            # Optional: Plot the signal
            plot_choice = input("\nPlot ECG signal? (y/n): ")
            if plot_choice.lower() == 'y':
                plt.figure(figsize=(12, 4))
                plt.plot(processed_ecg[:1200])  # Plot first 12 seconds
                plt.title(f'ECG Signal - Apnea Probability: {prob:.4f}')
                plt.xlabel('Sample')
                plt.ylabel('Normalized Amplitude')
                plt.grid(True, alpha=0.3)
                plt.show()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        collector.close()


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    
    MODEL_PATH = "./chestxray_best_model_1_val_loss=-0.1384.pt"
    SERIAL_PORT = "COM11"  
    
    print("🚀 ECG Apnea Detection System")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Serial Port: {SERIAL_PORT}")
    print("="*60 + "\n")
    
    run_ecg_inference(MODEL_PATH, SERIAL_PORT, device='cpu')