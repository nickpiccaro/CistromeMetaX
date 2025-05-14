from rapidfuzz import fuzz

print(fuzz.token_sort_ratio("epithelial", "epitheial"))