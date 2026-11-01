
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

np.random.seed(42)

N = 400_000  # So bit mo phong
EbN0_dB = np.arange(0, 25, 4)

# Tao so bit chia het cho 4
bits = np.random.randint(0, 2, N)
bits = bits[:len(bits) // 4 * 4]
bit_groups = bits.reshape(-1, 4)

# Anh xa Gray-coded 4-PAM tren moi truc I va Q
levels = np.array([-3, -1, 3, 1])

I = levels[2 * bit_groups[:, 0] + bit_groups[:, 1]]
Q = levels[2 * bit_groups[:, 2] + bit_groups[:, 3]]

# Chuan hoa nang luong trung binh symbol Es = 1
symbols = (I + 1j * Q) / np.sqrt(10)

ber_sim = []
ber_theory = []

for ebn0_db in EbN0_dB:
    ebn0 = 10 ** (ebn0_db / 10)

    # 16-QAM co 4 bit/symbol, Es = 1 nen Eb = 1/4
    # Moi truc I/Q co phuong sai nhieu N0/2
    noise_std = np.sqrt(1 / (8 * ebn0))
    noise = noise_std * (
        np.random.randn(len(symbols))
        + 1j * np.random.randn(len(symbols))
    )

    received = symbols + noise

    # Dua tin hieu ve thang do chua chuan hoa de quyet dinh
    rx_I = received.real * np.sqrt(10)
    rx_Q = received.imag * np.sqrt(10)

    # Tach muc 4-PAM tren moi truc
    def demap_4pam(x):
        out = np.zeros(len(x), dtype=int)
        out[(x >= -2) & (x < 0)] = 1
        out[(x >= 0) & (x < 2)] = 3
        out[x >= 2] = 2
        return out

    idx_I = demap_4pam(rx_I)
    idx_Q = demap_4pam(rx_Q)

    # Chuyen chi so muc thanh cap bit tuong ung
    decoded = np.column_stack((
        idx_I // 2, idx_I % 2,
        idx_Q // 2, idx_Q % 2
    )).reshape(-1)

    errors = np.sum(bits != decoded)
    ber_sim.append(errors / len(bits))

    # Xap xi BER ly thuyet Gray-coded 16-QAM tren AWGN
    M = 16
    k = np.log2(M)
    ber = (4 / k) * (1 - 1 / np.sqrt(M)) * 0.5 * erfc(
        np.sqrt(3 * k * ebn0 / (2 * (M - 1)))
    )
    ber_theory.append(ber)

print("Eb/N0 (dB) | BER mo phong | BER ly thuyet")
print("-" * 48)

for i, ebn0_db in enumerate(EbN0_dB):
    print(
        f"{ebn0_db:9.1f} | "
        f"{ber_sim[i]:12.6g} | "
        f"{ber_theory[i]:12.6g}"
    )

# Do thi BER
plt.figure(figsize=(9, 6))
plt.semilogy(EbN0_dB, ber_sim, "o-", label="16-QAM Simulation")
plt.semilogy(EbN0_dB, ber_theory, "s--", label="16-QAM Theory (approx.)")
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("16-QAM BER over AWGN")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()

# Chom sao 16-QAM
plt.figure(figsize=(7, 7))
plt.scatter(symbols.real, symbols.imag, s=5, alpha=0.35)
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.title("16-QAM Constellation")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()
