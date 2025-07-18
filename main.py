import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, ifftshift, fftfreq

# configs
N = 8192
T = 8.0

# setup
x = np.linspace(-T/2, T/2, N, endpoint=False)
dt = T / N 
xi = fftshift(fftfreq(N, d=dt))

# rectangular function
def get_rectangular_function():
    src = np.zeros_like(x)
    src[(x >= -0.5) & (x <= 0.5)] = 1.0
    tgt = np.sinc(xi)
    src_expr = r'$f(x) = \sqcap(x)$'
    prd_expr = r'$\hat{f}(\xi) = \mathrm{sinc}(\xi)$'
    tgt_expr = r'$g(\xi) = \mathrm{sinc}(\xi)$'
    return src, tgt, src_expr, prd_expr, tgt_expr, 'rectangular'

# gaussian function
def get_gaussian_function():
    src = np.exp(-np.pi * x**2)
    tgt = np.exp(-np.pi * xi**2)
    src_expr = r'$f(x) = e^{-\pi x^2}$'
    prd_expr = r'$\hat{f}(\xi) = e^{-\pi \xi^2}$'
    tgt_expr = r'$g(\xi) = e^{-\pi \xi^2}$'
    return src, tgt, src_expr, prd_expr, tgt_expr, 'gaussian'

# constant function
def get_constant_function():
    src = np.ones_like(x)
    tgt = np.zeros_like(xi)
    tgt[N // 2] = T
    src_expr = r'$f(x) = 1$'
    prd_expr = r'$\hat{f}(\xi) = \delta(\xi)$'
    tgt_expr = r'$g(\xi) = \delta(\xi)$'
    return src, tgt, src_expr, prd_expr, tgt_expr, 'constant'

# dirac delta function
def get_dirac_delta_function():
    src = np.zeros_like(x)
    src[N // 2] = 1.0 / dt
    tgt = np.ones_like(xi)
    src_expr = r'$f(x) = \delta(x)$'
    prd_expr = r'$\hat{f}(\xi) = 1$'
    tgt_expr = r'$g(\xi) = 1$'
    return src, tgt, src_expr, prd_expr, tgt_expr, 'dirac_delta'

# exponential function
def get_exponential_function():
    src = np.exp(2 * np.pi * 1j * x * 1.0)
    tgt = np.zeros_like(xi)
    tgt[xi == 1.0] = T
    src_expr = r'$f(x) = e^{2 \pi i x \xi_0}$'
    prd_expr = r'$\hat{f}(\xi) = \delta(\xi - \xi_0)$'
    tgt_expr = r'$g(\xi) = \delta(\xi - \xi_0)$'
    return src, tgt, src_expr, prd_expr, tgt_expr, 'exponential'

# shifted dirac delta function
def get_shifted_dirac_delta_function():
    src = np.zeros_like(x)
    src[x == 1.0] = 1.0 / dt
    tgt = np.exp(-2 * np.pi * 1j * xi * 1.0)
    src_expr = r'$f(x) = \delta(x - x_0)$'
    prd_expr = r'$\hat{f}(\xi) = e^{-2 \pi i x_0 \xi}$'
    tgt_expr = r'$g(\xi) = e^{-2 \pi i x_0 \xi}$'
    return src, tgt, src_expr, prd_expr, tgt_expr, 'shifted_dirac_delta'

# cosine function
def get_cosine_function():
    src = np.cos(2 * np.pi * x * 1.0)
    tgt = np.zeros_like(xi)
    tgt[xi == 1.0] = 0.5 * T
    tgt[xi == -1.0] = 0.5 * T
    src_expr = r'$f(x) = \cos(2 \pi x \xi_0)$'
    prd_expr = r'$\hat{f}(\xi) = \frac{1}{2} (\delta(\xi + \xi_0) + \delta(\xi - \xi_0))$'
    tgt_expr = r'$g(\xi) = \frac{1}{2} (\delta(\xi + \xi_0) + \delta(\xi - \xi_0))$'
    return src, tgt, src_expr, prd_expr, tgt_expr, 'cosine'

# sine function
def get_sine_function():
    src = np.sin(2 * np.pi * x * 1.0)
    tgt = np.zeros_like(xi, dtype=np.complex128)
    tgt[xi == 1.0] = -0.5j * T
    tgt[xi == -1.0] = 0.5j * T
    src_expr = r'$f(x) = \sin(2 \pi x \xi_0)$'
    prd_expr = r'$\hat{f}(\xi) = \frac{1}{2i} (\delta(\xi + \xi_0) - \delta(\xi - \xi_0))$'
    tgt_expr = r'$g(\xi) = \frac{1}{2i} (\delta(\xi + \xi_0) - \delta(\xi - \xi_0))$'
    return src, tgt, src_expr, prd_expr, tgt_expr, 'sine'

# dirac comb
def get_dirac_comb_function():
    src = np.zeros_like(x)
    src[np.isin(x, np.arange(-T//2, T//2 + 1, 1))] = 1.0 / dt
    tgt = np.zeros_like(xi)
    tgt[np.isin(xi, np.arange(-T//2, T//2 + 1, 1))] = T
    src_expr = r'$f(x) = \mathrm{III}(x)$'
    prd_expr = r'$\hat{f}(\xi) = \mathrm{III}(\xi)$'
    tgt_expr = r'$g(\xi) = \mathrm{III}(\xi)$'
    return src, tgt, src_expr, prd_expr, tgt_expr, 'dirac_comb'

def plot(src, tgt, src_expr, prd_expr, tgt_expr, save_prefix):
    prd = fftshift(fft(ifftshift(src))) * dt

    plt.figure(figsize=(9, 12))
    plt.subplot(4, 1, 1)
    plt.plot(x, src.real, 'b-', label=src_expr)
    plt.xlim(-T/2, T/2)
    if np.allclose(src.real, 0):
        plt.ylim(-0.05, 0.05)
    plt.xlabel(r'$x$ (s)')
    plt.ylabel(r'$y$ (a.u.)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title(r'$\mathrm{Re}[f](x)$')

    plt.subplot(4, 1, 2)
    plt.plot(x, src.imag, 'b-', label=src_expr)
    plt.xlim(-T/2, T/2)
    if np.allclose(src.imag, 0):
        plt.ylim(-0.05, 0.05)
    plt.xlabel(r'$x$ (s)')
    plt.ylabel(r'$y$ (a.u.)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title(r'$\mathrm{Im}[f](x)$')

    plt.subplot(4, 1, 3)
    plt.plot(xi, prd.real, 'b-', label=prd_expr)
    plt.plot(xi, tgt.real, 'r--', label=tgt_expr)
    plt.xlim(-T/2, T/2)
    if np.allclose(prd.real, 0):
        plt.ylim(-0.05, 0.05)
    plt.xlabel(r'$\xi$ (Hz)')
    plt.ylabel(r'$y$ (a.u.)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title(r'$\mathrm{Re}[\hat{f}](\xi)$')

    plt.subplot(4, 1, 4)
    plt.plot(xi, prd.imag, 'b-', label=prd_expr)
    plt.plot(xi, tgt.imag, 'r--', label=tgt_expr)
    plt.xlim(-T/2, T/2)
    if np.allclose(prd.imag, 0):
        plt.ylim(-0.05, 0.05)
    plt.xlabel(r'$\xi$ (Hz)')
    plt.ylabel(r'$y$ (a.u.)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title(r'$\mathrm{Im}[\hat{f}](\xi)$')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    functions = [
        get_rectangular_function,
        get_gaussian_function,
        get_constant_function,
        get_dirac_delta_function,
        get_exponential_function,
        get_shifted_dirac_delta_function,
        get_cosine_function,
        get_sine_function,
        get_dirac_comb_function
    ]

    for func in functions:
        src, tgt, src_expr, prd_expr, tgt_expr, save_prefix = func()
        plot(src, tgt, src_expr, prd_expr, tgt_expr, save_prefix)
