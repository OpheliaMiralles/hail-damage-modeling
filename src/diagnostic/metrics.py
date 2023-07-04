import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def log_spectral_distance_from_xarray(real_output, fake_output):
    epsilon = tf.keras.backend.epsilon()
    power_spectra_real = np.abs(np.fft.fft2(real_output)) ** 2
    power_spectra_fake = np.abs(np.fft.fft2(fake_output)) ** 2
    ratio = np.divide(power_spectra_real + epsilon, power_spectra_fake + epsilon)
    result = (10 * np.log10(ratio)) ** 2
    result = np.mean(result, axis=0)
    return result


def ks_stat_on_patch(patch1, patch2):
    points = np.linspace(-30., 30., 100)
    emp1 = tfp.distributions.Empirical(patch1)
    emp2 = tfp.distributions.Empirical(patch2)
    ks_stat = tf.reduce_max([tf.abs(emp1.cdf(p) - emp2.cdf(p)) for p in points], axis=0)
    return ks_stat

def spatially_convolved_ks_stat(real_output, fake_output, patch_size=3):
    to_concat = []
    patch_size = patch_size or fake_output.shape[2] // 10
    strides_apart = patch_size // 2
    i = 0
    for time in range(fake_output.shape[1]):
        for ch in range(fake_output.shape[-1]):
            print(f'Patch {i}/{fake_output.shape[1]*fake_output.shape[-1]}')
            patch1 = tf.image.extract_patches(real_output,
                                              sizes=(1, patch_size, patch_size, 1),
                                              strides=(1, strides_apart, strides_apart, 1),
                                              rates=(1, 1, 1, 1),
                                              padding='VALID')
            patch2 = tf.image.extract_patches(fake_output,
                                              sizes=(1, patch_size, patch_size, 1),
                                              strides=(1, strides_apart, strides_apart, 1),
                                              rates=(1, 1, 1, 1),
                                              padding='VALID')
            ks_stat_for_time_step = ks_stat_on_patch(patch1, patch2)
            to_concat.append(ks_stat_for_time_step)
            i+=1
    mean_ks_stat_img = tf.reduce_mean(to_concat, axis=(0, 1))
    return mean_ks_stat_img