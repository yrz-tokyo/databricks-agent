# Databricks notebook source
# MAGIC %md
# MAGIC # Intelligent Finance Agent - データ準備
# MAGIC
# MAGIC このノートブックは以下を実行します：
# MAGIC 1. PDF からテキストと単位情報を抽出
# MAGIC 2. チャンク分割（単位メタデータ付き）
# MAGIC 3. Delta Table に保存
# MAGIC 4. Vector Search インデックス作成

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 設定

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. ライブラリのインストールとインポート

# COMMAND ----------

# Install required libraries
%pip install pdfplumber tiktoken databricks-vectorsearch

# COMMAND ----------

# Restart Python to use newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

# Configuration (after Python restart)
CATALOG_NAME = "intelligent_finance_agent"
SCHEMA_NAME = "agent_schema"
VOLUME_NAME = "pdfs"
PDF_FILE_NAME = "kirin_2025_ir.pdf"
TABLE_NAME = "kirin_ir_chunks"

# Vector Search Configuration
VS_ENDPOINT_NAME = "finance_agent_endpoint"
VS_INDEX_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.kirin_ir_index"

# Chunking Configuration
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 200  # tokens

# Volume パス
VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}/{PDF_FILE_NAME}"

print(f"PDF Path: {VOLUME_PATH}")
print(f"Target Table: {CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}")
print(f"Vector Index: {VS_INDEX_NAME}")

# COMMAND ----------

import pdfplumber
import re
import tiktoken
from typing import List, Dict, Any, Optional
import pandas as pd
from pyspark.sql import SparkSession
from databricks.vector_search.client import VectorSearchClient
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. ユーティリティ関数

# COMMAND ----------

def extract_unit_from_page(text: str) -> Optional[str]:
    """
    ページテキストから単位情報を抽出する
    """
    unit_patterns = [
        r'単位[：:]\s*([百千万億兆]+円)',
        r'\(([百千万億兆]+円)\)',
        r'（([百千万億兆]+円)）',
        r'注[）)]\s*([百千万億兆]+円)',
    ]

    for pattern in unit_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return None


def chunk_text_by_tokens(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """テキストをトークン数でチャンク分割"""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        chunks = []

        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks
    except Exception as e:
        print(f"Tokenization failed, using character-based chunking: {e}")
        # Fallback to character-based chunking
        char_size = chunk_size * 4
        char_overlap = chunk_overlap * 4
        chunks = []
        for i in range(0, len(text), char_size - char_overlap):
            chunks.append(text[i:i + char_size])
        return chunks


def format_context_with_unit(context: str, metadata: Dict[str, Any]) -> str:
    """コンテキストに単位情報を追加"""
    unit = metadata.get("unit", "不明")
    page_num = metadata.get("page_number", "不明")
    return f"[ページ {page_num}, 単位: {unit}]\n{context}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. PDF からテキストと単位を抽出

# COMMAND ----------

print(f"Loading PDF from: {VOLUME_PATH}")

# Check if PDF exists
import os
if not os.path.exists(VOLUME_PATH):
    raise FileNotFoundError(f"PDF not found: {VOLUME_PATH}")

# Extract pages with units
pages_data = []

with pdfplumber.open(VOLUME_PATH) as pdf:
    print(f"Total pages: {len(pdf.pages)}")

    for page_num, page in enumerate(pdf.pages, start=1):
        # Extract text
        text = page.extract_text() or ""

        # Extract unit
        unit = extract_unit_from_page(text)

        pages_data.append({
            "page_number": page_num,
            "text": text,
            "unit": unit or "不明",
        })

        if page_num % 10 == 0:
            print(f"  Processed {page_num} pages...")

print(f"✓ Extracted {len(pages_data)} pages")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 単位の統計を確認

# COMMAND ----------

# Unit distribution
unit_counts = {}
for page in pages_data:
    unit = page["unit"]
    unit_counts[unit] = unit_counts.get(unit, 0) + 1

print("単位の分布:")
for unit, count in sorted(unit_counts.items(), key=lambda x: -x[1]):
    print(f"  {unit}: {count} ページ")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. チャンク分割（単位メタデータ付き）

# COMMAND ----------

chunks_data = []
chunk_id = 0

for page_data in pages_data:
    text = page_data["text"]
    page_num = page_data["page_number"]
    unit = page_data["unit"]

    # Skip empty pages
    if not text.strip():
        continue

    # Chunk text
    text_chunks = chunk_text_by_tokens(
        text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Add metadata to each chunk
    for i, chunk_text in enumerate(text_chunks):
        chunk_id += 1

        # Format with unit info
        formatted_text = format_context_with_unit(
            chunk_text,
            {"unit": unit, "page_number": page_num}
        )

        chunks_data.append({
            "chunk_id": f"chunk_{chunk_id}",
            "page_number": page_num,
            "chunk_index": i,
            "text": formatted_text,
            "original_text": chunk_text,
            "unit": unit,
            "source": PDF_FILE_NAME
        })

print(f"✓ Created {len(chunks_data)} chunks")

# Display sample
print("\nサンプルチャンク:")
print(chunks_data[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Delta Table に保存

# COMMAND ----------

# Create DataFrame
df = pd.DataFrame(chunks_data)
spark_df = spark.createDataFrame(df)

# Display schema
print("Schema:")
spark_df.printSchema()

# Display sample
display(spark_df.limit(5))

# COMMAND ----------

# Save to Delta Table
table_full_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}"

spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(table_full_name)

print(f"✓ Saved {len(chunks_data)} chunks to {table_full_name}")

# COMMAND ----------

# Verify table
result_df = spark.read.table(table_full_name)
print(f"Table row count: {result_df.count()}")
display(result_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Vector Search インデックス作成

# COMMAND ----------

# Initialize Vector Search Client
vsc = VectorSearchClient()

# Check/Create endpoint
try:
    endpoint = vsc.get_endpoint(VS_ENDPOINT_NAME)
    print(f"✓ Endpoint '{VS_ENDPOINT_NAME}' exists")
except Exception as e:
    print(f"Creating endpoint: {VS_ENDPOINT_NAME}")
    vsc.create_endpoint(
        name=VS_ENDPOINT_NAME,
        endpoint_type="STANDARD"
    )
    print(f"✓ Created endpoint: {VS_ENDPOINT_NAME}")

    # Wait for endpoint to be ready
    import time
    print("Waiting for endpoint to be ready...")
    time.sleep(60)

# COMMAND ----------

# Create Vector Search Index
embedding_endpoint = "databricks-gte-large-en"

print(f"Creating Vector Search index: {VS_INDEX_NAME}")
print(f"Embedding endpoint: {embedding_endpoint}")

try:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME,
        source_table_name=table_full_name,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="text",
        embedding_model_endpoint_name=embedding_endpoint
    )

    print(f"✓ Created Vector Search index: {VS_INDEX_NAME}")
except Exception as e:
    print(f"Index creation failed: {e}")
    print("Index may already exist or there may be a configuration issue")

# COMMAND ----------

# Trigger initial sync
print("Triggering initial sync...")
try:
    index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    index.sync()
    print("✓ Sync triggered")
except Exception as e:
    print(f"Sync trigger failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. 完了
# MAGIC
# MAGIC データ準備が完了しました！
# MAGIC
# MAGIC 次のステップ:
# MAGIC - Agent テストノートブック（`02_Test_Agent`）を実行
# MAGIC - Vector Search インデックスの同期が完了するまで数分待つ

# COMMAND ----------

print("="* 60)
print("データ準備完了!")
print("="* 60)
print(f"✓ Delta Table: {table_full_name}")
print(f"✓ Vector Index: {VS_INDEX_NAME}")
print(f"✓ Total chunks: {len(chunks_data)}")
print(f"✓ Pages processed: {len(pages_data)}")
