
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Dai Eb/N0
EbN0_dB = np.arange(0, 15, 2)
EbN0 = 10 ** (EbN0_dB / 10)

# BPSK va QPSK
ber_bpsk = 0.5 * erfc(np.sqrt(EbN0))
ber_qpsk = 0.5 * erfc(np.sqrt(EbN0))

# Coherent BFSK
ber_bfsk = 0.5 * erfc(np.sqrt(EbN0 / 2))

# BASK/OOK coherent detection
ber_bask = 0.5 * erfc(np.sqrt(EbN0 / 2))

# Gray-coded 16-QAM, cong thuc xap xi
M = 16
k = np.log2(M)
ber_16qam = (
    (4 / k) * (1 - 1 / np.sqrt(M))
    * 0.5
    * erfc(np.sqrt(3 * k * EbN0 / (2 * (M - 1))))
)

# Ve do thi so sanh
plt.figure(figsize=(10, 7))

plt.semilogy(EbN0_dB, ber_bask, "o-", label="BASK (coherent OOK)")
plt.semilogy(EbN0_dB, ber_bfsk, "s-", label="Coherent BFSK")
plt.semilogy(EbN0_dB, ber_bpsk, "^-", label="BPSK")
plt.semilogy(EbN0_dB, ber_qpsk, "d--", label="QPSK")
plt.semilogy(EbN0_dB, ber_16qam, "x-", label="16-QAM (approx.)")

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER Comparison over AWGN")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
