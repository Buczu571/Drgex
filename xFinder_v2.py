import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import resample

class VibrationAnalyzer:
    def __init__(self, volts_to_ms2=5):
        self.volts_to_ms2 = volts_to_ms2 

    def load_signal(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()
        self.sampling_rate = float(lines[0].strip())
        signal = np.array([float(l) for l in lines[1:] if l.strip()])
        return (signal - np.mean(signal)) 
    
    def compute_velocity_spectrum(self, signal):
        n = len(signal)
        window = np.hanning(n)
        yf = fft(signal * window)
        xf = fftfreq(n, 1/self.sampling_rate)[1:n//2]
        yf = (2.0 / np.sum(window)) * np.abs(yf[1:n//2])
        return xf, (yf / (2 * np.pi * xf)) * 1000 

    def find_1x_rpm(self, xf, velocity_spectrum, estimated_rpm):
        est_freq = estimated_rpm / 60.0

        search_bandwidth = 0.1 * est_freq
        mask = (xf > est_freq - search_bandwidth) & (xf < est_freq + search_bandwidth)
        if not np.any(mask):
            print("Brak danych w zakresie 1x RPM.")
            return None, None

        freq_range = xf[mask]
        spec_range = velocity_spectrum[mask]

        idx_max = np.argmax(spec_range)

        if 0 < idx_max < len(spec_range)-1:
            y0, y1, y2 = spec_range[idx_max-1], spec_range[idx_max], spec_range[idx_max+1]
            x0, x1, x2 = freq_range[idx_max-1], freq_range[idx_max], freq_range[idx_max+1]

            denominator = (y0 - 2*y1 + y2)
            if denominator == 0:
                peak_freq = x1
                peak_value = y1
            else:
                delta = 0.5 * (y0 - y2) / denominator
                peak_freq = x1 + delta * (x2 - x0) / 2
                peak_value = y1 - 0.25 * (y0 - y2) * delta
        else:
            peak_freq = freq_range[idx_max]
            peak_value = spec_range[idx_max]

        return peak_freq, peak_value

    def compute_order_spectrum(self, signal, rpm, orders=5, resample_factor=1, min_order=0.2):
        n = len(signal)
        window = np.hanning(n)
        signal_resampled = resample(signal * window, n * resample_factor)

        yf = fft(signal_resampled)
        xf = fftfreq(len(signal_resampled), 1/(self.sampling_rate * resample_factor))

        positive_mask = xf > 0
        xf = xf[positive_mask]
        yf = (2.0 / np.sum(window)) * np.abs(yf[positive_mask])  

        velocity_spectrum = (yf / (2 * np.pi * xf)) * 1000
        order_axis = xf / (rpm/60.0)

        order_mask = (order_axis <= orders) & (order_axis >= min_order)
        return order_axis[order_mask], velocity_spectrum[order_mask]
    
    def detect_faults_from_order(self, order_axis, order_spectrum, thresholds=None):
        """
        thresholds: dict zawierający np. {'misalignment_ratio': 0.5, 'looseness_level': 0.3}
        """
        if thresholds is None:
            thresholds = {'misalignment_ratio': 0.5, 'looseness_level': 0.3}

        result = []

        def get_order_amp(order):
            mask = (order_axis >= order - 0.05) & (order_axis <= order + 0.05)
            if np.any(mask):
                return np.max(order_spectrum[mask])
            return 0.0

        amp_1x = get_order_amp(1)
        amp_2x = get_order_amp(2)
        amp_3x = get_order_amp(3)
        amp_4x = get_order_amp(4)
        amp_5x = get_order_amp(5)

        if amp_1x > 3:
            result.append("Prawdopodobne niewyważenie (silna składowa 1x)")

        if amp_2x > thresholds['misalignment_ratio'] * amp_1x:
            result.append("Prawdopodobna niewspółosiowość (silna składowa 2x)")

        looseness_harmonics = [amp_3x, amp_4x, amp_5x]
        if all(h > thresholds['looseness_level'] * amp_1x for h in looseness_harmonics):
            result.append("Podejrzenie luzów mechanicznych (wiele harmonicznych)")

        return result
    
        
    def plot_order_spectrum(self, order_axis, order_spectrum, rpm, highlight_orders=None):
        plt.figure(figsize=(10,6))
        plt.plot(order_axis, order_spectrum)
        plt.xlabel('Order [xRPM]')
        plt.ylabel('Vibration velocity [mm/s]')
        plt.title(f'Order Spectrum (RPM = {rpm:.1f})')
        plt.grid(True)
        if highlight_orders:
            for o in highlight_orders:
                plt.axvline(x=o, color='r', linestyle='--', alpha=0.5)
        plt.show()

    def plot_order_waterfall(self, signals, rpms, labels=None, orders=10, resample_factor=1, min_order=0.2):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        num_signals = len(signals)
        y_ticks = []

        for i, (signal, rpm) in enumerate(zip(signals, rpms)):
            order_axis, order_spectrum = self.compute_order_spectrum(
                signal, rpm, orders=orders, resample_factor=resample_factor, min_order=min_order
            )
            z = np.full_like(order_axis, i)
            ax.plot(order_axis, z, order_spectrum)
            y_ticks.append(labels[i] if labels else str(i))

        ax.set_xlabel("Order [xRPM]")
        ax.set_ylabel("Sample")
        ax.set_zlabel("Vibration velocity [mm/s]")
        ax.set_title("Order Spectrum Waterfall")

        ax.set_yticks(range(num_signals))
        ax.set_yticklabels(y_ticks)

        plt.tight_layout()
        plt.show()

    def plot_fft_waterfall(self, signals, labels=None, max_freq=3000):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        num_signals = len(signals)
        y_ticks = []

        for i, signal in enumerate(signals):
            n = len(signal)
            window = np.hanning(n)
            windowed_signal = signal * window
            yf = fft(windowed_signal)
            xf = fftfreq(n, 1 / self.sampling_rate)

            xf = xf[1:n//2]
            yf = (2.0 / np.sum(window)) * np.abs(yf[1:n//2])

            mask = xf <= max_freq
            xf_cut = xf[mask]
            yf_cut = yf[mask]
            z = np.full_like(xf_cut, i)

            ax.plot(xf_cut, z, yf_cut)
            y_ticks.append(labels[i] if labels else str(i))

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Sample")
        ax.set_zlabel("Amplitude [V]")
        ax.set_title(f"FFT Waterfall Spectrum (0–{max_freq/1000:.1f} kHz)")

        ax.set_yticks(range(num_signals))
        ax.set_yticklabels(y_ticks)

        plt.tight_layout()
        plt.show()       
       
if __name__ == "__main__":
    analyzer = VibrationAnalyzer()
    signal = analyzer.load_signal('7.csv')
    estimated_rpm = 1220
    xf, vspec = analyzer.compute_velocity_spectrum(signal)
    peak_freq, peak_val = analyzer.find_1x_rpm(xf, vspec, estimated_rpm)
    print(f"Vibration velocity at 1xRPM: {peak_val:.2f} mm/s (frequency {peak_freq:.2f} Hz)")
    order_axis, order_spec = analyzer.compute_order_spectrum(signal, peak_freq*60, orders=10)
    analyzer.plot_order_spectrum(order_axis, order_spec, peak_freq*60, highlight_orders=[1,2,3])
    faults = analyzer.detect_faults_from_order(order_axis, order_spec)
    for f in faults:
        print("🛑", f)
         
'''
if __name__ == "__main__":
    analyzer = VibrationAnalyzer()

    filepaths = ['1.csv', '2.csv', '3.csv', '4.csv', '5.csv', '6.csv']
    labels = [f.replace('.csv', '') for f in filepaths]

    signals = []
    rpms = []

    for filepath, est_rpm in zip(filepaths, [1220, 1220, 1220, 1220, 1220, 1220]):
       signal = analyzer.load_signal(filepath)
       xf, vspec = analyzer.compute_velocity_spectrum(signal)
       peak_freq, _ = analyzer.find_1x_rpm(xf, vspec, est_rpm)
       rpm = peak_freq * 60
       signals.append(signal)
       rpms.append(rpm)

    #analyzer.plot_order_waterfall(signals, rpms, labels=labels, orders=20, min_order=0.2)
    #analyzer.plot_fft_waterfall(signals, labels=labels, max_freq=3000)  
'''