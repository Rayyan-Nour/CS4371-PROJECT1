"""Read text from topic1, extract named entities with spaCy, and publish
cumulative entity counts to topic2 as {"entity": ..., "count": ...}.

If --model-path is given, Member 3's classification query runs alongside
it as a second streaming query in the same application.

Example (inside the Spark container):
  spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
    project/entity_stream.py \
    --checkpoint checkpoints/entities \
    --model-path models/ag_news_lr \
    --class-checkpoint checkpoints/classes
"""
import argparse
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_json, struct, to_json, udf
from pyspark.sql.types import ArrayType, StringType, StructField, StructType


# Entity types we keep. Dates, numbers, money, etc. are dropped because
# they say little about what the source is talking about.
KEEP_LABELS = {"PERSON", "ORG", "GPE", "NORP", "EVENT", "PRODUCT"}

MESSAGE_SCHEMA = StructType([
    StructField("id", StringType()),
    StructField("text", StringType()),
    StructField("source", StringType()),
    StructField("collected_at", StringType()),
])

_nlp = None
_spacy_model = "en_core_web_sm"


def get_nlp():
    # Loaded once per Python worker process, not once per message
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load(_spacy_model, disable=["parser", "lemmatizer"])
    return _nlp


def normalize(name):
    name = re.sub(r"https?://\S+", "", name)
    name = re.sub(r"\s+", " ", name).strip(" \t\n#@\"'.,:;!?()[]{}")
    name = re.sub(r"^(the|The|THE) ", "", name)
    name = re.sub(r"('s|’s)$", "", name)
    if len(name) < 2 or not re.search(r"[A-Za-z]", name):
        return None
    # Times and readings like "8:52PM MST" or "Vent:0.0 m/s" from bot posts
    if re.search(r"\d[:.]\d", name):
        return None
    # "microsoft" and "Microsoft" should count as the same entity
    if name.islower():
        name = name.title()
    return name


def extract_entities(text):
    if not text:
        return []
    doc = get_nlp()(text[:5000])
    entities = []
    for ent in doc.ents:
        if ent.label_ in KEEP_LABELS:
            name = normalize(ent.text)
            if name:
                entities.append(name)
    return entities


def parse_args():
    parser = argparse.ArgumentParser(description="Named-entity counts from topic1 to topic2")
    parser.add_argument("--bootstrap", default="kafka:29092")
    parser.add_argument("--input-topic", default="topic1")
    parser.add_argument("--output-topic", default="topic2")
    parser.add_argument("--checkpoint", required=True,
                        help="checkpoint directory for the entity-count query")
    parser.add_argument("--starting-offsets", default="latest",
                        help="'latest' or 'earliest'")
    parser.add_argument("--trigger", default="10 seconds")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--model-path",
                        help="saved PipelineModel; enables the classification query")
    parser.add_argument("--class-checkpoint",
                        help="checkpoint directory for the classification query")
    return parser.parse_args()


def main():
    global _spacy_model
    args = parse_args()
    if args.model_path and not args.class_checkpoint:
        raise SystemExit("--class-checkpoint is required when --model-path is set")
    _spacy_model = args.spacy_model

    spark = SparkSession.builder.appName("EntityStream").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    raw = (spark.readStream
           .format("kafka")
           .option("kafka.bootstrap.servers", args.bootstrap)
           .option("subscribe", args.input_topic)
           .option("startingOffsets", args.starting_offsets)
           .option("failOnDataLoss", "false")
           .load())

    messages = (raw
                .select(from_json(col("value").cast("string"), MESSAGE_SCHEMA).alias("m"))
                .select("m.*")
                .filter(col("text").isNotNull()))

    entities_udf = udf(extract_entities, ArrayType(StringType()))
    counts = (messages
              .select(explode(entities_udf(col("text"))).alias("entity"))
              .groupBy("entity")
              .count())

    def publish_counts(batch_df, batch_id):
        # In update mode each batch holds only entities whose totals changed.
        # Cache so the count below doesn't re-run spaCy on the batch.
        batch_df.persist()
        (batch_df
         .select(col("entity").alias("key"),
                 to_json(struct("entity", "count")).alias("value"))
         .write
         .format("kafka")
         .option("kafka.bootstrap.servers", args.bootstrap)
         .option("topic", args.output_topic)
         .save())
        print(f"Batch {batch_id}: published {batch_df.count()} entity counts")
        batch_df.unpersist()

    entity_query = (counts.writeStream
                    .outputMode("update")
                    .foreachBatch(publish_counts)
                    .option("checkpointLocation", args.checkpoint)
                    .trigger(processingTime=args.trigger)
                    .queryName("entity_counts")
                    .start())
    print("Entity query started")

    if args.model_path:
        # Member 3's helper; expected to start and return its own query
        from classification_stream import start_classification_query
        start_classification_query(spark, messages, args.model_path,
                                   args.class_checkpoint, args.trigger)
        print("Classification query started")

    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("\nStopping queries...")
    finally:
        for query in spark.streams.active:
            query.stop()
        spark.stop()
        print("Stopped cleanly.")


if __name__ == "__main__":
    main()
