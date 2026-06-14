from openai import OpenAI
import csv
import time
import os

# Try to get API key from environment variable first, fallback to hardcoded
API_KEY = os.getenv("OPENAI_API_KEY", "<fallback_api_key>")
MODEL_NAME = "gpt-5-mini"

if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it or update the API_KEY in this script.")

client = OpenAI(api_key=API_KEY)

def _save_progress(rows, filename):
    """Save progress incrementally"""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["query_id", "query_text", "topic", "response_text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Test API key with a simple request
print("Testing API key...")
try:
    test_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "test"}],
        max_completion_tokens=5
    )
    print(f"✓ API key is valid. Using model: {MODEL_NAME}")
except Exception as e:
    print(f"✗ Error with API key: {e}")
    print("\nPossible issues:")
    print("1. API key expired or invalid - regenerate it at https://platform.openai.com/api-keys")
    print("2. Insufficient credits - check your OpenAI account balance")
    print("3. API key doesn't have correct permissions")
    raise

# 1) Load your 60 queries from queries.csv
queries = []
with open("queries.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        queries.append(row)

print(f"Loaded {len(queries)} queries")

# Check if we have existing responses to resume from
existing_responses = {}
output_file = "responses.csv"
if os.path.exists(output_file):
    print(f"Found existing {output_file}, checking for completed queries...")
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_responses[row["query_id"]] = row
    print(f"Found {len(existing_responses)} already completed queries")

# 2) Call OpenAI for each query and store responses
output_rows = list(existing_responses.values()) if existing_responses else []

for q in queries:
    # Skip if already processed
    if q["query_id"] in existing_responses:
        print(f"Skipping query {q['query_id']} (already processed)")
        continue
    prompt = q["query_text"]
    qid = q["query_id"]

    print(f"Running query {qid}: {prompt[:60]}...")

    max_retries = 3
    retry_count = 0
    success = False
    
    while retry_count < max_retries and not success:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
            )

            # Response text
            text = response.choices[0].message.content or ""

            output_rows.append({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "topic": q["topic"],
                "response_text": text.replace("\n", " ").strip()
            })
            success = True
            
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            error_type = type(e).__name__
            error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            
            # Check for rate limiting (429) or resource exhausted errors
            is_rate_limit = (
                "429" in error_msg or 
                "rate_limit" in error_type.lower() or
                "rate limit" in error_msg.lower() or
                "quota" in error_msg.lower() or
                error_code == 429
            )
            
            # Check for authentication errors
            is_auth_error = (
                "authentication" in error_msg.lower() or
                "invalid_api_key" in error_type.lower() or
                "401" in error_msg or
                error_code == 401
            )
            
            # If we've successfully processed queries, auth error might be a rate limit in disguise
            if is_auth_error and len(output_rows) > 0:
                # If we got past the first query, this might be rate limiting, not an invalid key
                print(f"\n⚠ API key error after {len(output_rows)} successful queries - might be rate limiting")
                if retry_count < max_retries:
                    wait_time = min(60, 15 * retry_count)  # 15s, 30s, 45s
                    print(f"   Retrying after {wait_time}s (retry {retry_count}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"\n✗ Persistent API key error after {max_retries} retries")
                    print(f"   Error: {error_msg}")
                    print(f"\n   This might be:")
                    print(f"   1. Rate limiting (most likely if it worked before)")
                    print(f"   2. API key actually invalid or expired")
                    print(f"\n   Solutions:")
                    print(f"   - Wait a few minutes and rerun the script (it will resume)")
                    print(f"   - Check API key at https://platform.openai.com/api-keys")
                    _save_progress(output_rows, output_file)
                    raise
            elif is_auth_error:
                # Failed on first query - likely a real API key issue
                print(f"\n✗ ERROR: API key issue detected!")
                print(f"   Error: {error_msg}")
                print(f"\n   Solutions:")
                print(f"   1. Go to https://platform.openai.com/api-keys")
                print(f"   2. Regenerate your API key")
                print(f"   3. Make sure you have sufficient credits")
                print(f"   4. Update the API_KEY in this script or set OPENAI_API_KEY env variable")
                raise  # Don't retry on API key errors for first query
            
            elif is_rate_limit:
                # Rate limit - use longer backoff
                if retry_count < max_retries:
                    wait_time = min(60, 10 * retry_count)  # 10s, 20s, 30s (max 60s)
                    print(f"   ⚠ Rate limit hit! Waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"\n✗ Rate limited after {max_retries} retries. Saving progress and stopping.")
                    print(f"   You can rerun this script to resume from query {qid}")
                    _save_progress(output_rows, output_file)
                    raise  # Stop execution on persistent rate limits
            
            elif retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff: 2s, 4s, 8s
                print(f"   Retry {retry_count}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n✗ Failed after {max_retries} retries: {error_msg}")
                # Save error response instead of crashing
                output_rows.append({
                    "query_id": q["query_id"],
                    "query_text": q["query_text"],
                    "topic": q["topic"],
                    "response_text": f"ERROR: {error_msg}"
                })

    # Save progress after each successful query (incremental save)
    if success:
        _save_progress(output_rows, output_file)
    
    # Longer pause to avoid rate limits
    time.sleep(1.0)

# 3) Final save (already saved incrementally, but save once more to be sure)
_save_progress(output_rows, output_file)

print(f"\n✓ Done! Processed {len(output_rows)}/{len(queries)} queries")
print(f"  Saved to {output_file}")
