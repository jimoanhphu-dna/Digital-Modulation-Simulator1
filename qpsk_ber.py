
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Cau hinh mo phong
np.random.seed(42)
N = 100_000
EbN0_dB = np.arange(0, 11, 2)

# Tao bit ngau nhien
bits = np.random.randint(0, 2, N)

# QPSK can so bit chan
bits = bits[:len(bits) // 2 * 2]

# Gom 2 bit thanh 1 symbol
bit_pairs = bits.reshape(-1, 2)

# Anh xa bit thanh truc I va Q: 0 -> -1, 1 -> +1
I = 2 * bit_pairs[:, 0] - 1
Q = 2 * bit_pairs[:, 1] - 1

# Chuan hoa nang luong symbol Es = 1
symbols = (I + 1j * Q) / np.sqrt(2)

ber_sim = []
ber_theory = []

for ebn0_db in EbN0_dB:
    ebn0 = 10 ** (ebn0_db / 10)

    # QPSK AWGN: moi nhanh I/Q co phuong sai N0/2
    noise_std = np.sqrt(1 / (4 * ebn0))
    noise = noise_std * (
        np.random.randn(len(symbols))
        + 1j * np.random.randn(len(symbols))
    )

    received = symbols + noise

    # Giai dieu che bang nguong 0 tren I va Q
    detected_I = (received.real > 0).astype(int)
    detected_Q = (received.imag > 0).astype(int)

    detected_bits = np.column_stack(
        (detected_I, detected_Q)
    ).reshape(-1)

    errors = np.sum(bits != detected_bits)
    ber_sim.append(errors / len(bits))

    # BER ly thuyet QPSK tren AWGN
    ber_theory.append(
        0.5 * erfc(np.sqrt(ebn0))
    )

# In ket qua
print("Eb/N0 (dB) | BER mo phong | BER ly thuyet")
print("-" * 48)

for i, ebn0_db in enumerate(EbN0_dB):
    print(
        f"{ebn0_db:9.1f} | "
        f"{ber_sim[i]:12.6g} | "
        f"{ber_theory[i]:12.6g}"
    )

# Ve do thi BER
plt.figure(figsize=(9, 6))
plt.semilogy(
    EbN0_dB, ber_sim, "o-",
    label="QPSK Simulation"
)
plt.semilogy(
    EbN0_dB, ber_theory, "s--",
    label="QPSK Theory"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("QPSK BER over AWGN")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()

# Ve chum sao QPSK
plt.figure(figsize=(6, 6))
plt.scatter(
    symbols.real[:2000],
    symbols.imag[:2000],
    s=12,
    label="Transmitted symbols"
)
plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.title("QPSK Constellation")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()
