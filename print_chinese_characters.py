def print_all_chinese_chars():
    # Define ranges as tuples (start, end, description)
    ranges = [
        (0x4E00, 0x9FFF, "CJK Unified Ideographs"),
        (0x3400, 0x4DBF, "CJK Unified Ideographs Extension A"),
        (0x20000, 0x2A6DF, "CJK Unified Ideographs Extension B"),
        (0x2A700, 0x2B73F, "CJK Unified Ideographs Extension C"),
        (0x2B740, 0x2B81F, "CJK Unified Ideographs Extension D"),
        (0x2B820, 0x2CEAF, "CJK Unified Ideographs Extension E"),
        (0xF900, 0xFAFF, "CJK Compatibility Ideographs")
    ]

    for start, end, desc in ranges:
        print(f"\n{desc}")
        print("=" * 50)
        print(f"Range: U+{start:X} to U+{end:X}")
        print("Characters:")

        try:
            chars = []
            for code_point in range(start, end + 1):
                char = chr(code_point)
                chars.append(char)

                # Print in chunks of 50 characters to avoid overwhelming the terminal
                if len(chars) == 50:
                    print(''.join(chars))
                    chars = []

            if chars:  # Print any remaining characters
                print(''.join(chars))

            print(f"Total characters in range: {end - start + 1}")

        except Exception as e:
            print(f"Error processing range: {e}")

        print("-" * 50)

if __name__ == "__main__":
    print("Printing all Chinese characters in defined ranges...")
    print("Note: Some characters may not display correctly depending on your terminal font support")
    print_all_chinese_chars()