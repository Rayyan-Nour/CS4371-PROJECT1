"""Second streaming query: classify live topic1 messages with the saved
AG News model and print a running count per predicted class at each trigger.

Imported by entity_stream.py, which passes in its parsed topic1 stream:
  from classification_stream import start_classification_query
"""
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def start_classification_query(spark, messages, model_path, checkpoint, trigger="10 seconds"):
    model = PipelineModel.load(model_path)

    per_class = (model.transform(messages.select("text"))
                 .groupBy(col("prediction").cast("int").alias("prediction"))
                 .count())

    def print_counts(batch_df, batch_id):
        counts = {r["prediction"]: r["count"] for r in batch_df.collect()}
        total = sum(counts.values())
        print(f"\n--- Class counts, batch {batch_id} ({total} messages so far) ---")
        for i, name in enumerate(CLASS_NAMES):
            n = counts.get(i, 0)
            share = 100 * n / total if total else 0
            print(f"  {name:<9} {n:>7}  ({share:.1f}%)")

    return (per_class.writeStream
            .outputMode("complete")
            .foreachBatch(print_counts)
            .option("checkpointLocation", checkpoint)
            .trigger(processingTime=trigger)
            .queryName("class_counts")
            .start())
