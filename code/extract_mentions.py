import csv
import re

# Define regex patterns for each brand (case-insensitive)
brand_patterns = {
    "1Password": re.compile(r"\b1 ?password\b", re.IGNORECASE),
    "Bitwarden": re.compile(r"\bbit ?warden\b", re.IGNORECASE),
    "LastPass": re.compile(r"\blast ?pass\b", re.IGNORECASE),
    "Dashlane": re.compile(r"\bdash ?lane\b", re.IGNORECASE),
    "Keeper": re.compile(r"\bkeeper\b", re.IGNORECASE),
    "NordPass": re.compile(r"\bnord ?pass\b", re.IGNORECASE),
    "RoboForm": re.compile(r"\brobo ?form\b", re.IGNORECASE),
}

# 1) Load the responses
responses = []
with open("responses.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        responses.append(row)

print(f"Loaded {len(responses)} responses")

# 2) Build the long-format mentions table
mentions_rows = []

for row in responses:
    qid = row["query_id"]
    text = row["response_text"]

    for brand, pattern in brand_patterns.items():
        mention = 1 if pattern.search(text) else 0

        mentions_rows.append({
            "query_id": qid,
            "brand": brand,
            "mention": mention
        })

# 3) Save mentions.csv
with open("mentions.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["query_id", "brand", "mention"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(mentions_rows)

print("Done. Saved mentions to mentions.csv")
