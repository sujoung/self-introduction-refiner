from server import word_stats

stats = word_stats("Eh, jag heter Anna, liksom, och jag jobbar typ som utvecklare. Eh.")
assert stats["word_count"] == 12, stats
assert stats["fillers"] == {"eh": 2, "liksom": 1, "typ": 1}, stats
assert stats["filler_count"] == 4, stats
print("ok")
