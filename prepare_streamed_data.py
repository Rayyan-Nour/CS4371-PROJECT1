from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim


spark = SparkSession.builder.appName("PrepareStreamedData").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


df = spark.read.json("data/streamed_messages.jsonl")
original_count = df.count()

# Remove empty or whitespace-only text
df_no_empty = df.filter(col("text").isNotNull() & (trim(col("text")) != ""))
after_empty_count = df_no_empty.count()

# Remove duplicate texts
df_clean = df_no_empty.dropDuplicates(["text"])
final_count = df_clean.count()

# Print counts for your report
print("Original count:", original_count)
print("Empty removed:", original_count - after_empty_count)
print("Duplicates removed:", after_empty_count - final_count)
print("Final count:", final_count)

# Save cleaned dataset for Member 3
df_clean.write.mode("overwrite").parquet("data/cleaned_messages.parquet")

spark.stop()