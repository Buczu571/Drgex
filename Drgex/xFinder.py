import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import resample



def gaussian(x, amp, mean, std): return amp * np.exp(-(x - mean)**2 / (2 * std**2))

class VibrationAnalyzer:
    def __init__(self, volts_to_ms2=5.0):
        self.volts_to_ms2 = volts_to_ms2 

    def load_signal(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()
        self.sampling_rate = float(lines[0].strip())
        signal = np.array([float(l) for l in lines[1:] if l.strip()])
        return (signal - np.mean(signal)) * self.volts_to_ms2
    
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

    def compute_order_spectrum(self, signal, rpm, orders=5, resample_factor=1):
        n = len(signal)
        window = np.hanning(n)
        signal_resampled = resample(signal * window, n * resample_factor)
        yf = fft(signal_resampled)
        xf = fftfreq(n * resample_factor, 1/(self.sampling_rate * resample_factor))
        xf, yf = xf[xf > 0], (2.0 / np.sum(window)) * np.abs(yf[xf > 0])
        velocity_spectrum = (yf / (2 * np.pi * xf)) * 1000
        order_axis = xf / (rpm/60.0)
        mask = order_axis <= orders
        return order_axis[mask], velocity_spectrum[mask]

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


    def extract_1x_region(self, order_axis, order_spectrum, center=1.0, width=0.1, n_points=64, global_max=150):
        mask = (order_axis >= center - width/2) & (order_axis <= center + width/2)
        if not np.any(mask):
            print("Brak danych w zakresie 1x order.")
            return None
        region = np.interp(
            np.linspace(center - width/2, center + width/2, n_points),
            order_axis[mask],
            order_spectrum[mask]
        )
        region /= global_max
        region = np.clip(region, 0, 1) 
        return region


    def export_1x_region(self, filepath_in, filepath_out, estimated_rpm, center=1.0, width=0.1, n_points=64, global_max=150):
        signal = self.load_signal(filepath_in)
        xf, vspec = self.compute_velocity_spectrum(signal)
        peak_freq, _ = self.find_1x_rpm(xf, vspec, estimated_rpm)
        if peak_freq is None:
            return
        order_axis, order_spec = self.compute_order_spectrum(signal, peak_freq * 60)
        region = self.extract_1x_region(order_axis, order_spec, center=center, width=width, n_points=n_points, global_max=global_max)
        if region is not None:
            np.save(filepath_out, region)
            print(f"Zapisano region 1x do: {filepath_out}")


if __name__ == "__main__":
    analyzer = VibrationAnalyzer()
    signal = analyzer.load_signal('1.csv')
    estimated_rpm = 1200
    xf, vspec = analyzer.compute_velocity_spectrum(signal)
    peak_freq, peak_val = analyzer.find_1x_rpm(xf, vspec, estimated_rpm)
    print(f"Vibration velocity at 1xRPM: {peak_val:.2f} mm/s (frequency {peak_freq:.2f} Hz)")
    order_axis, order_spec = analyzer.compute_order_spectrum(signal, peak_freq*60, orders=10)
    analyzer.plot_order_spectrum(order_axis, order_spec, peak_freq*60, highlight_orders=[1,2,3])
    #analyzer.export_1x_region("zdrowa.csv", "zdrowa.npy", estimated_rpm=1220)