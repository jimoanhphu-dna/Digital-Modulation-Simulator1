
import numpy as np
from scipy.special import erfc


# ==========================================
# 1. DIGITAL MODULATION
# ==========================================

def modulate(name, bits):
    bits = np.asarray(bits, dtype=int)

    if name == "BASK":
        return bits.astype(float).astype(complex)

    if name == "BFSK":
        return bits.astype(complex)

    if name == "BPSK":
        return (2 * bits - 1).astype(complex)

    if name == "QPSK":
        if len(bits) % 2 != 0:
            raise ValueError("QPSK requires an even number of bits.")

        b = bits.reshape(-1, 2)
        return ((2*b[:, 0]-1) + 1j*(2*b[:, 1]-1)) / np.sqrt(2)

    if name == "16-QAM":
        if len(bits) % 4 != 0:
            raise ValueError("16-QAM requires bits divisible by 4.")

        b = bits.reshape(-1, 4)
        levels = np.array([-3, -1, 3, 1])

        I = levels[2*b[:, 0] + b[:, 1]]
        Q = levels[2*b[:, 2] + b[:, 3]]

        return (I + 1j*Q) / np.sqrt(10)

    raise ValueError("Unsupported modulation technique.")


# ==========================================
# 2. DIGITAL DEMODULATION
# ==========================================

def demodulate(name, received):
    received = np.asarray(received)

    if name == "BASK":
        return (received.real > np.sqrt(2)/2).astype(int)

    if name == "BFSK":
        return (received.real > 0).astype(int)

    if name == "BPSK":
        return (received.real > 0).astype(int)

    if name == "QPSK":
        out = np.column_stack((
            (received.real > 0).astype(int),
            (received.imag > 0).astype(int)
        ))
        return out.reshape(-1)

    if name == "16-QAM":
        x = received.real * np.sqrt(10)
        y = received.imag * np.sqrt(10)

        def pam_bits(v):
            idx = np.zeros(len(v), dtype=int)
            idx[(v >= -2) & (v < 0)] = 1
            idx[(v >= 0) & (v < 2)] = 3
            idx[v >= 2] = 2
            return np.column_stack((idx // 2, idx % 2))

        return np.column_stack((
            pam_bits(x),
            pam_bits(y)
        )).reshape(-1)

    raise ValueError("Unsupported demodulation technique.")


# ==========================================
# 3. AWGN CHANNEL
# ==========================================

def add_awgn(name, symbols, ebn0_db):
    ebn0 = 10 ** (ebn0_db / 10)

    bits_per_symbol = {
        "BASK": 1,
        "BFSK": 1,
        "BPSK": 1,
        "QPSK": 2,
        "16-QAM": 4
    }[name]

    if name == "BASK":
        tx = symbols * np.sqrt(2)
    else:
        tx = symbols

    noise_std = np.sqrt(1 / (2 * bits_per_symbol * ebn0))

    noise = noise_std * (
        np.random.randn(len(tx))
        + 1j * np.random.randn(len(tx))
    )

    return tx + noise


# ==========================================
# 4. THEORETICAL BER
# ==========================================

def theoretical_ber(name, ebn0):
    if name in ("BPSK", "QPSK"):
        return 0.5 * erfc(np.sqrt(ebn0))

    if name in ("BFSK", "BASK"):
        return 0.5 * erfc(np.sqrt(ebn0 / 2))

    if name == "16-QAM":
        M = 16
        k = np.log2(M)

        return (
            (4/k)
            * (1 - 1/np.sqrt(M))
            * 0.5
            * erfc(np.sqrt(3*k*ebn0/(2*(M-1))))
        )

    raise ValueError("Unsupported modulation technique.")
# =========================
# BFSK PASSBAND MODULATION
# =========================

FS = 1000
SYMBOL_RATE = 100
SPS = FS // SYMBOL_RATE


def modulate_bfsk_wave(bits):
    bits = np.asarray(bits, dtype=int)
    t = np.arange(SPS) / FS
    waves = []

    for bit in bits:
        f = 200 if bit == 1 else 100
        wave = np.cos(2 * np.pi * f * t)
        wave = wave / np.sqrt(np.sum(wave**2))
        waves.extend(wave)

    return np.asarray(waves)


def demodulate_bfsk_wave(rx_wave, n_symbols):
    t = np.arange(SPS) / FS

    ref0 = np.cos(2 * np.pi * 100 * t)
    ref1 = np.cos(2 * np.pi * 200 * t)

    detected = []

    for i in range(n_symbols):
        segment = rx_wave[i*SPS:(i+1)*SPS]

        corr0 = np.sum(segment * ref0)
        corr1 = np.sum(segment * ref1)

        detected.append(1 if corr1 > corr0 else 0)

    return np.asarray(detected, dtype=int)


def add_awgn_bfsk(tx_wave, ebn0_db):
    ebn0 = 10 ** (ebn0_db / 10)
    noise_std = np.sqrt(1 / (2 * ebn0))
    noise = noise_std * np.random.randn(len(tx_wave))

    return tx_wave + noise