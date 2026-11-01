import numpy as np
from modulation import (
    modulate_bfsk_wave,
    demodulate_bfsk_wave,
    add_awgn_bfsk
)

np.random.seed(42)

# Tạo chuỗi bit kiểm thử
bits = np.random.randint(0, 2, 100_000)

# Điều chế BFSK
tx_wave = modulate_bfsk_wave(bits)

# Kiểm tra giải điều chế khi không có nhiễu
detected = demodulate_bfsk_wave(tx_wave, len(bits))
errors = np.sum(bits != detected)

print("BFSK MODULE TEST")
print("------------------------------")
print(f"Number of bits : {len(bits)}")
print(f"Waveform length: {len(tx_wave)}")
print(f"Errors         : {errors}")

# Kiểm tra thêm nhiễu tại Eb/N0 = 6 dB
rx_wave = add_awgn_bfsk(tx_wave, 6)
detected_noise = demodulate_bfsk_wave(rx_wave, len(bits))
errors_noise = np.sum(bits != detected_noise)

print(f"Errors with noise (6 dB): {errors_noise}")
print("BFSK MODULE TEST COMPLETED!")