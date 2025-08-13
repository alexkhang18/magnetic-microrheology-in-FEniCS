import numpy as np

def test(arr):

    # Check the shape of the array
    try:
        assert arr.shape == (2, 3)
        print("Shape check passed: Array shape is correct.")
    except AssertionError:
        print(f"Shape check failed: Expected (2, 3), got {arr.shape}.")

    # Check the data type of the array
    try:
        assert arr.dtype == np.int64
        print("Data type check passed: Array data type is correct.")
    except AssertionError:
        print(f"Data type check failed: Expected int64, got {arr.dtype}.")

    # Check the values in the array
    expected_values = np.array([[1, 2, 3], [4, 5, 6]])
    try:
        assert np.array_equal(arr, expected_values)
        print("Values check passed: Array values are correct.")
    except AssertionError:
        print("Values check failed: Array values do not match.")

# Create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6] , [4, 5, 6]])
test(arr.astype(np.int32))

# Create a 2D array with the correct shape and data type
arr_correct = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
test(arr_correct)

# Create a 2D array with the correct shape but incorrect data type
arr_wrong_dtype = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
test(arr_wrong_dtype)

# Create a 2D array with the correct shape but incorrect values
arr_wrong_values = np.array([[1, 2, 3], [4, 5, 7]], dtype=np.int64)
test(arr_wrong_values)

# Create a 2D array with the correct shape but extra rows
arr_extra_rows = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
test(arr_extra_rows)

# Create a 2D array with the correct shape but extra columns
arr_extra_columns = np.array([[1, 2, 3, 7], [4, 5, 6, 8]], dtype=np.int64)
test(arr_extra_columns)

# Create a 2D array with the correct shape but mixed data types
arr_mixed_types = np.array([[1, 2, 3], [4, '5', 6]], dtype=object)
test(arr_mixed_types)

# Create a 2D array with the correct shape but with NaN values
arr_nan_values = np.array([[1, 2, 3], [4, np.nan, 6]], dtype=np.float64)
test(arr_nan_values)

# Create a 2D array with the correct shape but with inf values
arr_inf_values = np.array([[1, 2, 3], [4, np.inf, 6]], dtype=np.float64)
test(arr_inf_values)

# Create a 2D array with the correct shape but with negative values
arr_negative_values = np.array([[1, 2, 3], [-4, -5, -6]], dtype=np.int64)
test(arr_negative_values)

# Create a 2D array with the correct shape but with large values
arr_large_values = np.array([[1, 2, 3], [4000000000, 5000000000, 6000000000]], dtype=np.int64)
test(arr_large_values)

# Create a 2D array with the correct shape but with small values
arr_small_values = np.array([[1, 2, 3], [0, 0, 0]], dtype=np.int64)
test(arr_small_values)