
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Cau hinh mo phong
np.random.seed(42)
N = 100_000
EbN0_dB = np.arange(0, 11, 2)

# Hai bit 0 va 1 co nang luong bang nhau
bits = np.random.randint(0, 2, N)

# BER mo phong va ly thuyet
ber_sim = []
ber_theory = []

# Mo phong coherent BFSK
for ebn0_db in EbN0_dB:
    ebn0 = 10 ** (ebn0_db / 10)

    # Muc tin hieu ±sqrt(Eb), voi Eb = 1
    # Nhiễu Gaussian tren truc quyet dinh
    noise_std = np.sqrt(1 / ebn0)

    # Hieu hai dau ra tuong quan:
    # coherent BFSK co khoang cach Euclidean tuong ung sqrt(2*Eb)
    # Mo hinh quy doi ve quyet dinh nhi phan
    decision_signal = np.where(bits == 1, 1.0, -1.0)
    noise = np.random.normal(0, noise_std, N)
    received = decision_signal + noise

    detected_bits = (received > 0).astype(int)

    errors = np.sum(bits != detected_bits)
    ber_sim.append(errors / N)

    # BER ly thuyet coherent BFSK
    ber_theory.append(
        0.5 * erfc(np.sqrt(ebn0 / 2))
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

# Ve do thi
plt.figure(figsize=(9, 6))
plt.semilogy(
    EbN0_dB, ber_sim, "o-",
    label="Coherent BFSK Simulation"
)
plt.semilogy(
    EbN0_dB, ber_theory, "s--",
    label="Coherent BFSK Theory"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("Coherent BFSK BER over AWGN")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
