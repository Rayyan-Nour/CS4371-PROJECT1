"""Train a topic classifier on AG News and save it as a PipelineModel.

Example (inside the Spark container):
  spark-submit project/train_model.py \
    --train data/ag_news/train.parquet \
    --test data/ag_news/test.parquet \
    --model-path models/ag_news_lr
"""
import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IDF, HashingTF, RegexTokenizer, StopWordsRemover
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train TF-IDF + LogisticRegression on AG News")
    parser.add_argument("--train", required=True, help="AG News training Parquet file")
    parser.add_argument("--test", help="optional AG News test Parquet file for accuracy")
    parser.add_argument("--model-path", required=True, help="where to save the PipelineModel")
    parser.add_argument("--num-features", type=int, default=1 << 18)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--reg-param", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    spark = SparkSession.builder.appName("TrainAGNews").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    train = (spark.read.parquet(args.train)
             .select("text", col("label").cast("double").alias("label")))

    print(f"Training articles: {train.count()}")
    for row in train.groupBy("label").count().orderBy("label").collect():
        print(f"  {int(row['label'])} {CLASS_NAMES[int(row['label'])]:<9} {row['count']}")

    # Lowercase words only; punctuation and digits are split away
    tokenizer = RegexTokenizer(inputCol="text", outputCol="words",
                               pattern="[^a-z]+", toLowercase=True, minTokenLength=2)
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    tf = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=args.num_features)
    idf = IDF(inputCol="tf", outputCol="features", minDocFreq=2)
    lr = LogisticRegression(featuresCol="features", labelCol="label",
                            maxIter=args.max_iter, regParam=args.reg_param,
                            family="multinomial")

    model = Pipeline(stages=[tokenizer, remover, tf, idf, lr]).fit(train)
    model.write().overwrite().save(args.model_path)
    print(f"Saved PipelineModel to {args.model_path}")

    if args.test:
        test = (spark.read.parquet(args.test)
                .select("text", col("label").cast("double").alias("label")))
        predictions = model.transform(test)
        accuracy = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        ).evaluate(predictions)
        print(f"AG News test accuracy: {accuracy:.4f} on {test.count()} articles")

    spark.stop()


if __name__ == "__main__":
    main()
