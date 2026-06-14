import pandas as pd
import numpy as np

# ===========================
# CONFIG
# ===========================
RESPONSES_CHATGPT_FILE = "data/responses_chatgpt.csv"
RESPONSES_GEMINI_FILE = "data/responses_gemini.csv"
MENTIONS_CHATGPT_FILE = "data/mentions_chatgpt.csv"
MENTIONS_GEMINI_FILE = "data/mentions_gemini.csv"
BRAND_FEATURES_FILE = "data/brand_features.csv"
TOPIC_HITS_FILE = "data/topic_brand_hits.csv"
OUT_FILE = "data/dataset.csv"

REVIEW_SOURCE = "g2"

def main():
    # 1) Load data
    responses_chatgpt = pd.read_csv(RESPONSES_CHATGPT_FILE)
    responses_gemini = pd.read_csv(RESPONSES_GEMINI_FILE)
    mentions_chatgpt = pd.read_csv(MENTIONS_CHATGPT_FILE)
    mentions_gemini = pd.read_csv(MENTIONS_GEMINI_FILE)
    brand_features = pd.read_csv(BRAND_FEATURES_FILE)
    topic_hits = pd.read_csv(TOPIC_HITS_FILE)

    # Basic sanity prints
    print("responses_chatgpt:", responses_chatgpt.shape)
    print("responses_gemini:", responses_gemini.shape)
    print("mentions_chatgpt:", mentions_chatgpt.shape)
    print("mentions_gemini:", mentions_gemini.shape)
    print("brand_features:", brand_features.shape)
    print("topic_hits:", topic_hits.shape)

    # 2) Process ChatGPT data
    df_chatgpt = mentions_chatgpt.merge(
        responses_chatgpt[["query_id", "query_text", "topic", "response_text"]],
        on="query_id",
        how="left"
    )
    df_chatgpt["source"] = "chatgpt"

    # 3) Process Gemini data
    df_gemini = mentions_gemini.merge(
        responses_gemini[["query_id", "query_text", "topic", "response_text"]],
        on="query_id",
        how="left"
    )
    df_gemini["source"] = "gemini"

    # 4) Combine both datasets
    df = pd.concat([df_chatgpt, df_gemini], ignore_index=True)

    # 5) Merge brand-level features (Trustpilot + G2 + lighthouse)
    df = df.merge(
        brand_features,
        on="brand",
        how="left"
    )
    
    # Convert rating columns from comma decimal format (e.g., "4,5") to float (4.5)
    rating_cols = ["avgrating_b_tp", "avgrating_b_g2"]
    for col in rating_cols:
        if col in df.columns:
            # Replace comma with period and convert to float
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
    
    # Ensure review count columns are numeric
    review_cols = ["reviewcount_b_tp", "reviewcount_b_g2"]
    for col in review_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 6) Merge brand × topic hits
    df = df.merge(
        topic_hits,
        on=["brand", "topic"],
        how="left"
    )
    
    # Ensure topic hit columns are numeric
    topic_hit_cols = ["listicle_topic_hits_bt", "reddit_topic_hits_bt", 
                      "youtube_topic_hits_bt", "linkedin_topic_hits_bt", "domain_topic_hits_bt"]
    for col in topic_hit_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 7) (Optional) sort rows and reorder columns nicely
    # Sort by query_id, source, then brand for readability
    df = df.sort_values(by=["query_id", "source", "brand"]).reset_index(drop=True)

    # Desired column order
    desired_cols = [
        "query_id",
        "source",
        "query_text",
        "topic",
        "response_text",
        "brand",
        "mention",
        "avgrating_b_tp",
        "reviewcount_b_tp",
        "avgrating_b_g2",
        "reviewcount_b_g2",
        "lighthouse_seo_b",
        "listicle_topic_hits_bt",
        "reddit_topic_hits_bt",
        "youtube_topic_hits_bt",
        "linkedin_topic_hits_bt",
        "domain_topic_hits_bt",
    ]

    # Keep only columns that actually exist (in case names differ)
    cols_present = [c for c in desired_cols if c in df.columns]
    df = df[cols_present]

    # 8) Convert numeric columns back to comma format for Google Sheets compatibility
    def float_to_comma_format(series):
        """Convert float series to comma-separated decimal format"""
        return series.apply(lambda x: str(x).replace(".", ",") if pd.notna(x) else "")
    
    # Convert rating columns (float) back to comma format
    rating_cols = ["avgrating_b_tp", "avgrating_b_g2"]
    for col in rating_cols:
        if col in df.columns:
            df[col] = float_to_comma_format(df[col])
    
    # Convert reviewindex columns (float) back to comma format
    reviewindex_cols = ["reviewcount_b_tp", "reviewcount_b_g2"]
    for col in reviewindex_cols:
        if col in df.columns:
            df[col] = float_to_comma_format(df[col])

    # 9) Save
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"Saved {OUT_FILE} with {len(df)} rows and {len(df.columns)} columns.")


if __name__ == "__main__":
    main()
