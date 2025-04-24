from rapidfuzz import fuzz

print(fuzz.ratio("thisisatest", "thisisanewtest"))

print(fuzz.ratio("testathisis", "thisisanewtest"))