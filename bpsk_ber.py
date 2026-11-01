
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Cau hinh mo phong
np.random.seed(42)
N = 100_000
EbN0_dB = np.arange(0, 11, 2)

ber_sim = []

# Mo phong BPSK qua kenh AWGN
bits = np.random.randint(0, 2, N)
symbols = 2 * bits - 1

for ebn0 in EbN0_dB:
    # Chuyen Eb/N0 tu dB sang dang tuyen tinh
    ebn0_linear = 10 ** (ebn0 / 10)

    # Do lech chuan nhieu cho BPSK voi Eb = 1
    noise_std = np.sqrt(1 / (2 * ebn0_linear))
    noise = np.random.normal(0, noise_std, N)

    # Tin hieu thu va giai dieu che
    received = symbols + noise
    detected_bits = (received > 0).astype(int)

    # Tinh BER thuc nghiem
    errors = np.sum(bits != detected_bits)
    ber_sim.append(errors / N)

# BER ly thuyet cua BPSK
ebn0_linear = 10 ** (EbN0_dB / 10)
ber_theory = 0.5 * erfc(np.sqrt(ebn0_linear))

# In bang ket qua
print("Eb/N0 (dB) | BER mo phong | BER ly thuyet")
print("-" * 48)

for i in range(len(EbN0_dB)):
    print(
        f"{EbN0_dB[i]:9.1f} | "
        f"{ber_sim[i]:12.6g} | "
        f"{ber_theory[i]:12.6g}"
    )

# Ve do thi BER
plt.figure(figsize=(9, 6))
plt.semilogy(
    EbN0_dB, ber_sim, "o-",
    label="BPSK Simulation"
)
plt.semilogy(
    EbN0_dB, ber_theory, "s--",
    label="BPSK Theory"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BPSK BER over AWGN Channel")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
