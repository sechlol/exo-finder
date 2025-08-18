LC_WINDOW_SIZE = 2**11  # 2048 points, or about 2.84 days per light curve
DATASET_LENGTH = 2**17  # 131_072 examples

# LC_WINDOW_SIZE = 2**12  # 4096 points, or about 5.68 days per light curve
# DATASET_LENGTH = 2**16  # 65_536 examples

LC_WINDOW_MIN_SIZE = int(LC_WINDOW_SIZE * 0.8)
