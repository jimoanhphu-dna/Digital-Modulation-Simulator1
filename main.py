
import numpy as np
import matplotlib.pyplot as plt

# Tao chuoi bit
np.random.seed(42)
N = 20
bits = np.random.randint(0, 2, N)

# Dieu che BPSK: bit 0 -> -1, bit 1 -> +1
symbols = 2 * bits - 1

# Them nhieu AWGN
noise_std = 0.6
noise = np.random.normal(0, noise_std, N)
received = symbols + noise

# Ve do thi
plt.figure(figsize=(12, 5))

plt.subplot(2, 1, 1)
plt.stem(symbols)
plt.title("BPSK Transmitted Symbols")
plt.ylabel("Amplitude")
plt.yticks([-1, 1])
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(received)
plt.axhline(0, linestyle="--")
plt.title("BPSK Received Symbols with AWGN")
plt.xlabel("Symbol Index")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()

# Buoc 5: Giai dieu che BPSK
detected_bits = (received > 0).astype(int)

# Buoc 6: Dem so bit loi
errors = np.sum(bits != detected_bits)
ber = errors / N

print("\nBit goc:")
print(bits)

print("\nBit thu duoc:")
print(detected_bits)

print("\nSo bit loi:", errors, "/", N)
print("BER =", ber)

print("Bits ban dau:")
print(bits)

print("\nTin hieu phat:")
print(symbols)

print("\nTin hieu thu sau khi co nhieu:")
print(np.round(received, 3))
