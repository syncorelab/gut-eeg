from pathlib import Path
import numpy as np
import scipy.io
import h5py


def load_mat_smart(mat_path: str):
    """
    Load a MATLAB .mat file.
    Supports:
      - standard .mat files via scipy.io.loadmat
      - v7.3 / HDF5 .mat files via h5py
    """
    mat_path = Path(mat_path)

    try:
        data = scipy.io.loadmat(mat_path)
        data = {k: v for k, v in data.items() if not k.startswith("__")}
        print("Loaded with scipy.io.loadmat")
        return data
    except NotImplementedError:
        print("Detected MATLAB v7.3 / HDF5 file, loading with h5py...")
        out = {}
        with h5py.File(mat_path, "r") as f:
            for key in f.keys():
                out[key] = np.array(f[key])
        return out


def preview_array(arr: np.ndarray, max_items: int = 10):
    """
    Print a small preview of an array depending on dimensionality.
    """
    arr = np.array(arr)

    print(f"    shape: {arr.shape}")
    print(f"    dtype: {arr.dtype}")

    if arr.ndim == 0:
        print(f"    value: {arr}")
    elif arr.ndim == 1:
        print(f"    first {min(max_items, len(arr))} values:")
        print(f"    {arr[:max_items]}")
    elif arr.ndim == 2:
        r = min(5, arr.shape[0])
        c = min(5, arr.shape[1])
        print(f"    first {r} x {c} block:")
        print(arr[:r, :c])
    elif arr.ndim == 3:
        a = min(3, arr.shape[0])
        b = min(3, arr.shape[1])
        c = min(3, arr.shape[2])
        print(f"    first {a} x {b} x {c} block:")
        print(arr[:a, :b, :c])
    else:
        flat = arr.flatten()
        print(f"    first {min(max_items, len(flat))} flattened values:")
        print(f"    {flat[:max_items]}")


def inspect_subject_mat(mat_path: str):
    """
    Open a subject-specific .mat file and inspect all top-level variables.
    """
    data = load_mat_smart(mat_path)

    print("\n" + "=" * 70)
    print(f"INSPECTING FILE: {mat_path}")
    print("=" * 70)

    if not data:
        print("No variables found.")
        return

    print("\nVariables found:")
    for key in data.keys():
        print(f"  - {key}")

    print("\nDetailed inspection:")
    for key, value in data.items():
        print("\n" + "-" * 70)
        print(f"Variable: {key}")
        print(f"Type: {type(value)}")

        if isinstance(value, np.ndarray):
            preview_array(value)
        else:
            print(f"    value: {value}")


if __name__ == "__main__":
    # Change this path to your subject-specific .mat file
    mat_file = r"/home/gutproject/Desktop/guteeg/gut-eeg/data/processed_sub_withICA/014_B4_c1ref.mat"

    inspect_subject_mat(mat_file)