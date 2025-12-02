from PIL import Image
import numpy as np

def image_to_c_array(image_path, var_name, out_file):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    H, W, C = arr.shape

    with open(out_file, "w") as f:
        f.write(f"const int IMAGE_HEIGHT = {H};\n")
        f.write(f"const int IMAGE_WIDTH = {W};\n\n")

        f.write(f"const uint8_t {var_name}[{H*W}][3] = {{\n")
        for y in range(H):
            for x in range(W):
                r, g, b = arr[y, x]
                f.write(f"    {{{r},{g},{b}}},\n")
        f.write("};\n")


if __name__ == "__main__":
    for name in ["2018", "3063", "5096", "6046", "8068"]:
        image_to_c_array(
            f"testcases/{name}.jpg",
            f"im_{name}",
            f"testcases/im_{name}.h"
        )