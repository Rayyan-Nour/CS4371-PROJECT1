"""Classify every cleaned streamed message with the saved AG News model.

Writes one CSV with the text, predicted class and probability vector for
every message, plus a summary with counts per class and 5 examples each.

Example (inside the Spark container):
  spark-submit project/classify_batch.py \
    --input data/cleaned_messages.parquet \
    --model-path models/ag_news_lr \
    --output-dir outputs/batch_predictions \
    --summary outputs/batch_summary.txt
"""
import argparse

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import (array_distinct, array_max, col, concat_ws, filter as array_filter,
                                   format_number, lower, rand, size, split, transform, when)

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
EXAMPLES_PER_CLASS = 5
EXAMPLE_SEED = 4371

# Rules for choosing the example messages shown in the summary (the CSV
# still contains every message). Without them the most confident predictions
# are bot posts that repeat one word ("DING DING DING ...") and explicit posts.
MIN_EXAMPLE_PROBABILITY = 0.7
MIN_WORDS, MIN_DISTINCT_RATIO, MAX_MENTIONS = 6, 0.7, 2
EXPLICIT_WORDS = r"\b(fuck\w*|shit\w*|nsfw|porn\w*|sex\w*|nude\w*|cock|cum|penis|pussy|dick|slut\w*|horny|vore\w*|onlyfans|ass|asses|bitch\w*|wtf|damn|kink\w*|fetish\w*|lewd\w*|forcefe\w*|\w*belly\w*|\w*tummy\w*|stuffing)\b"


def class_name(prediction_col):
    expr = None
    for i, name in enumerate(CLASS_NAMES):
        expr = when(prediction_col == i, name) if expr is None else expr.when(prediction_col == i, name)
    return expr


def parse_args():
    parser = argparse.ArgumentParser(description="Batch predictions for streamed messages")
    parser.add_argument("--input", required=True, help="cleaned messages (Parquet)")
    parser.add_argument("--model-path", required=True, help="saved PipelineModel")
    parser.add_argument("--output-dir", required=True, help="directory for the predictions CSV")
    parser.add_argument("--summary", required=True, help="text file for counts and examples")
    return parser.parse_args()


def main():
    args = parse_args()
    spark = SparkSession.builder.appName("ClassifyBatch").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    messages = spark.read.parquet(args.input).select("id", "text")
    model = PipelineModel.load(args.model_path)

    probs = vector_to_array(col("probability"))
    predictions = (model.transform(messages)
                   .select("id", "text",
                           col("prediction").cast("int").alias("predicted_label"),
                           class_name(col("prediction")).alias("predicted_class"),
                           probs.alias("probs"))
                   .withColumn("top_probability", array_max("probs"))
                   .cache())

    # The probability vector is written in class order: World, Sports, Business, Sci/Tech
    (predictions
     .select("id", "text", "predicted_label", "predicted_class",
             *[col("probs")[i].alias(f"prob_{n.lower().replace('/', '_')}")
               for i, n in enumerate(CLASS_NAMES)],
             concat_ws(",", transform("probs", lambda p: format_number(p, 4))).alias("probability"))
     .coalesce(1)
     .write.mode("overwrite").option("header", True).option("escape", '"')
     .csv(args.output_dir))

    total = predictions.count()
    counts = {r["predicted_class"]: r["count"]
              for r in predictions.groupBy("predicted_class").count().collect()}

    lines = [f"Messages classified: {total}", "", "Messages per predicted class:"]
    for name in CLASS_NAMES:
        n = counts.get(name, 0)
        lines.append(f"  {name:<9} {n:>7}  ({100 * n / total:.1f}%)")

    # Example pool: confident predictions on readable messages
    words = array_filter(split(lower(col("text")), "[^a-z]+"), lambda w: w != "")
    readable = (predictions
                .withColumn("n_words", size(words))
                .withColumn("n_distinct", size(array_distinct(words)))
                .filter(col("top_probability") >= MIN_EXAMPLE_PROBABILITY)
                .filter(col("n_words") >= MIN_WORDS)
                .filter(col("n_distinct") >= MIN_DISTINCT_RATIO * col("n_words"))
                .filter(size(split(col("text"), "@")) - 1 <= MAX_MENTIONS)
                .filter(~lower(col("text")).rlike(EXPLICIT_WORDS)))
    lines += ["", f"Examples: {EXAMPLES_PER_CLASS} random messages per class (seed {EXAMPLE_SEED}) with",
              f"predicted probability >= {MIN_EXAMPLE_PROBABILITY}, at least {MIN_WORDS} words, no heavy word",
              f"repetition (spam bots), at most {MAX_MENTIONS} @mentions, and no explicit language."]

    for i, name in enumerate(CLASS_NAMES):
        examples = (readable.filter(col("predicted_label") == i)
                    .orderBy(rand(EXAMPLE_SEED))
                    .limit(EXAMPLES_PER_CLASS)
                    .collect())
        lines += ["", f"=== {name}: {len(examples)} examples ==="]
        if len(examples) < EXAMPLES_PER_CLASS:
            lines.append(f"  Only {len(examples)} messages were predicted as {name}.")
        for row in examples:
            text = " ".join(row["text"].split())
            prob_list = ", ".join(f"{p:.3f}" for p in row["probs"])
            lines.append(f"  [P({name})={row['top_probability']:.3f}]  {text[:200]}")
            lines.append(f"      probability vector [World, Sports, Business, Sci/Tech] = [{prob_list}]")

    summary = "\n".join(lines)
    print(summary)
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\nPredictions CSV written to {args.output_dir}, summary to {args.summary}")

    spark.stop()


if __name__ == "__main__":
    main()
