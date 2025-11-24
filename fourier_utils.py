import numpy as np
from scipy.signal import find_peaks
from scipy import integrate
import standard_data_utils as stand_utils


def _simpson(y, x):
    """Safely integrate using Simpson's rule, compatible across SciPy versions."""
    if len(y) < 2:
        return 0.0
    try:
        return integrate.simpson(y, x=x)
    except AttributeError:
        return integrate.simps(y, x=x)

def find_first_harmonic(fft_data, fft_freq_data, prominence=None, minimum_height=1e-8):
    """
    Find the first and second harmonic frequencies and amplitudes in FFT data.

    Args:
        fft_data (np.array): FFT magnitude data
        fft_freq_data (np.array): Frequency values corresponding to FFT data
        prominence (float, optional): Prominence for peak detection. If None, it's calculated automatically.
        minimum_height (float): Minimum peak height to consider

    Returns:
        tuple: First harmonic frequency, amplitude, and second harmonic frequency
    """
    if prominence is None:
        peak1_index, _ = find_peaks(fft_data, height=minimum_height)
        if peak1_index.size == 0:
            return 0.0, 0.0, 0.0
        peak1_amp = np.max(fft_data[peak1_index])
        prominence = peak1_amp / 7 if peak1_amp != 0 else minimum_height

    peaks, _ = find_peaks(fft_data, height=minimum_height, prominence=prominence)
    peak_amplitudes = fft_data[peaks]
    sorted_peak_indices = np.argsort(peak_amplitudes)[::-1]

    if len(sorted_peak_indices) == 0:
        return 0.0, 0.0, 0.0

    first_harmonic_index = peaks[sorted_peak_indices[0]]
    first_harmonic_amplitude = fft_data[first_harmonic_index]
    first_harmonic_frequency = fft_freq_data[first_harmonic_index]

    if len(sorted_peak_indices) > 1:
        second_harmonic_index = peaks[sorted_peak_indices[1]]
        second_harmonic_frequency = fft_freq_data[second_harmonic_index]
    else:
        second_harmonic_frequency = 0.0

    print(f"First harmonic: {first_harmonic_amplitude} at {first_harmonic_frequency}")
    print(f"Second harmonic frequency: {second_harmonic_frequency}")

    return first_harmonic_frequency, first_harmonic_amplitude, second_harmonic_frequency

def integrate_higher_modes(ft_data, freq_vals, first_harmonic_freq, harmonic_fraction=1):
    """
    Integrate the higher modes in FFT data.

    Args:
        ft_data (np.array): FFT magnitude data
        freq_vals (np.array): Frequency values
        first_harmonic_freq (float): Frequency of the first harmonic
        harmonic_fraction (float): Fraction of harmonic difference to use for integration width

    Returns:
        float: Integrated area of higher modes
    """
    first_harmonic_idx = np.argmin(np.abs(freq_vals - first_harmonic_freq))
    second_harmonic_freq = 2 * first_harmonic_freq
    second_harmonic_idx = np.argmin(np.abs(freq_vals - second_harmonic_freq))

    freq_diff = freq_vals[second_harmonic_idx] - freq_vals[first_harmonic_idx]
    integration_width = freq_diff * harmonic_fraction
    first_harmonic_upper_limit = first_harmonic_freq + integration_width / 2
    upper_idx = np.searchsorted(freq_vals, first_harmonic_upper_limit)

    area_higher_modes = _simpson(ft_data[upper_idx:], freq_vals[upper_idx:]) * 2
    return area_higher_modes

def integrate_first_harmonic_fwhm(ft_data, freq_vals, first_harmonic_freq, width_factor=2):
    """
    Integrate the first harmonic using FWHM method.

    Args:
        ft_data (np.array): FFT magnitude data
        freq_vals (np.array): Frequency values
        first_harmonic_freq (float): Frequency of the first harmonic
        width_factor (float): Factor to multiply FWHM for integration width

    Returns:
        tuple: Integrated area and FWHM of the first harmonic
    """
    peak_idx = np.argmin(np.abs(freq_vals - first_harmonic_freq))
    peak_value = ft_data[peak_idx]
    half_max = peak_value / 2

    left_idx = np.where(ft_data[:peak_idx] <= half_max)[0][-1] if np.any(ft_data[:peak_idx] <= half_max) else 0
    right_idx = np.where(ft_data[peak_idx:] <= half_max)[0][0] + peak_idx if np.any(ft_data[peak_idx:] <= half_max) else len(ft_data) - 1

    fwhm = freq_vals[right_idx] - freq_vals[left_idx]
    integration_width = fwhm * width_factor
    lower_limit = first_harmonic_freq - integration_width / 2
    upper_limit = first_harmonic_freq + integration_width / 2

    lower_idx = np.searchsorted(freq_vals, lower_limit)
    upper_idx = np.searchsorted(freq_vals, upper_limit)

    area = _simpson(ft_data[lower_idx:upper_idx], freq_vals[lower_idx:upper_idx])
    return area, fwhm

def integrate_first_harmonic(ft_data, freq_vals, first_harmonic_freq, second_harmonic_freq, harmonic_fraction=1):
    """
    Integrate the first harmonic.

    Args:
        ft_data (np.array): FFT magnitude data
        freq_vals (np.array): Frequency values
        first_harmonic_freq (float): Frequency of the first harmonic
        second_harmonic_freq (float): Frequency of the second harmonic
        harmonic_fraction (float): Fraction of harmonic difference to use for integration width

    Returns:
        float: Integrated area of the first harmonic
    """
    first_harmonic_idx = np.argmin(np.abs(freq_vals - first_harmonic_freq))
    second_harmonic_idx = np.argmin(np.abs(freq_vals - second_harmonic_freq))

    freq_diff = np.abs(freq_vals[second_harmonic_idx] - freq_vals[first_harmonic_idx])

    right_integration_width = freq_diff * harmonic_fraction
    left_integration_width = first_harmonic_freq * harmonic_fraction * 0.9

    lower_limit = first_harmonic_freq - left_integration_width / 2
    upper_limit = first_harmonic_freq + right_integration_width / 2
    print(f"Integration limits: {lower_limit}, {upper_limit}")

    lower_idx = np.argmin(np.abs(freq_vals - lower_limit))
    upper_idx = np.argmin(np.abs(freq_vals - upper_limit))

    area_slice = ft_data[lower_idx:upper_idx]
    freq_slice = freq_vals[lower_idx:upper_idx]
    area = _simpson(area_slice, freq_slice) * 2
    return area

def analyse_fourier_data(sorted_files, freq_vals, norm_factor, is_temporal_ft, cut_index=None, time_index_strategy="center_max"):
    """
    Analyze Fourier data from a list of files.

    Args:
        sorted_files (list): List of file paths to analyze
        freq_vals (np.array): Frequency values
        norm_factor (float): Normalization factor
        is_temporal_ft (bool): Whether the data is temporal Fourier transform
        cut_index (int, optional): Index for temporal cut
        time_index_strategy (str): Strategy for selecting the time index when using spatial data.
            "center_max" follows the original behaviour (use the row with the largest value at the spatial centre).
            "tracked_peak_max" follows the tracked peak over time (requires cut_index).

    Returns:
        dict: Dictionary containing analysis results
    """
    if is_temporal_ft and cut_index is None:
        raise ValueError("You must specify a cut index for temporal Fourier transform data")

    first_mode_ft_peaks_amp = []
    first_mode_ft_peaks_freq = []
    first_mode_ft_peak_area = []
    higher_modes_ft_peak_area = []

    freq_vals = np.abs(freq_vals[:len(freq_vals)//2])

    for file in sorted_files:
        data = np.loadtxt(file)

        if is_temporal_ft:
            psi_cut_vals = stand_utils.find_temporal_cut_of_x_peaks(data, cut_index)
            t_vals = data[:, 0]
            freq_vals = np.fft.fftfreq(len(t_vals), np.diff(t_vals)[0])
            freq_vals = np.abs(freq_vals[:len(freq_vals)//2])
        else:
            if time_index_strategy == "center_max":
                max_index = np.argmax(data[:, data.shape[1] // 2])
            elif time_index_strategy == "tracked_peak_max":
                if cut_index is None:
                    raise ValueError("You must specify cut_index when using the 'tracked_peak_max' strategy.")
                temporal_cut = stand_utils.find_temporal_cut_of_x_peaks(data, cut_index)
                max_index = int(np.argmax(temporal_cut))
            else:
                raise ValueError(f"Unknown time_index_strategy '{time_index_strategy}'.")
            psi_cut_vals = data[max_index, 1:]

        fft_psi_vals = np.abs(np.fft.fft(psi_cut_vals, norm="forward")[:len(psi_cut_vals)//2]) / norm_factor

        first_harmonic_frequency, first_harmonic_amplitude, second_harmonic_frequency = find_first_harmonic(fft_psi_vals, freq_vals)
        first_mode_ft_peaks_amp.append(first_harmonic_amplitude)
        first_mode_ft_peaks_freq.append(first_harmonic_frequency)

        first_mode_area = integrate_first_harmonic(fft_psi_vals, freq_vals, first_harmonic_frequency, second_harmonic_frequency)
        first_mode_ft_peak_area.append(first_mode_area)

        higher_mode_area = integrate_higher_modes(fft_psi_vals, freq_vals, first_harmonic_frequency)
        higher_modes_ft_peak_area.append(higher_mode_area)

    return {
        'first_mode_ft_peaks_amp': first_mode_ft_peaks_amp,
        'first_mode_ft_peaks_freq': first_mode_ft_peaks_freq,
        'first_mode_ft_peak_area': first_mode_ft_peak_area,
        'higher_modes_ft_peak_area': higher_modes_ft_peak_area
    }



def calculate_nqc_amplitude(psi_vals, num_crit, norm_density=True):
    """
    Calculate |n^{q_c}| — the modulus of the density Fourier component at the critical wavenumber.

    Args:
        psi_vals (np.ndarray): 1D complex array of the wavefunction ψ(x)
        num_crit (float): Number of critical wavelengths in the simulation domain
                          (from the simulation input)
        norm_density (bool): If True, normalize so the mean density <|ψ|²> = 1

    Returns:
        float: |n^{q_c}|
    """
    N = len(psi_vals)
    L = 2 * np.pi * num_crit       # Domain length in code units
    dx = L / N

    # Density
    rho = np.abs(psi_vals)**2

    # Normalize the average density to 1 if required
    if norm_density:
        rho /= np.mean(rho)

    # Compute FFT of the density
    rho_k = np.fft.fft(rho)

    # Build the k-grid (same scaling as simulation)
    k_vals = np.fft.fftfreq(N, d=dx) * 2 * np.pi

    # Find the FFT bin closest to k = +1 (critical mode)
    idx_crit = np.argmin(np.abs(k_vals - 1.0))

    # Convert discrete FFT to continuum-normalized coefficient:
    # n(k) ≈ (1/L) ∫ ρ(x) e^{-ikx} dx  ≈ (dx / L) * sum_j ρ_j e^{-ikx_j}
    n_qc = (dx / L) * rho_k[idx_crit]
    return np.abs(n_qc)
