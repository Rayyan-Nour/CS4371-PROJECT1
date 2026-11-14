"""Hand-label evaluation of the AG News model on streamed messages.

Step 1 - export 50 random messages (fixed seed) for manual labelling.
The sheet has no predictions, so labels are chosen without seeing them:
  spark-submit project/evaluate.py sample \
    --input data/cleaned_messages.parquet --sheet outputs/hand_labels.csv

Fill the manual_label column with World, Sports, Business, Sci/Tech or
None of these.

Step 2 - compare the hand labels with the saved model's predictions:
  spark-submit project/evaluate.py score \
    --sheet outputs/hand_labels.csv --model-path models/ag_news_lr \
    --results outputs/evaluation_results.csv --summary outputs/evaluation_summary.txt
"""
import argparse
import csv

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NONE_LABEL = "None of these"
LABEL_ALIASES = {
    "world": "World", "sports": "Sports", "sport": "Sports", "business": "Business",
    "sci/tech": "Sci/Tech", "scitech": "Sci/Tech", "sci-tech": "Sci/Tech", "sci tech": "Sci/Tech",
    "none of these": NONE_LABEL, "none": NONE_LABEL,
}


def parse_args():
    parser = argparse.ArgumentParser(description="50-message hand-label evaluation")
    sub = parser.add_subparsers(dest="step", required=True)

    sample = sub.add_parser("sample", help="export random messages for manual labels")
    sample.add_argument("--input", required=True, help="cleaned messages (Parquet)")
    sample.add_argument("--sheet", required=True, help="CSV to write for labelling")
    sample.add_argument("--n", type=int, default=50)
    sample.add_argument("--seed", type=int, default=4371)

    score = sub.add_parser("score", help="compare manual labels with predictions")
    score.add_argument("--sheet", required=True, help="labelled CSV from the sample step")
    score.add_argument("--model-path", required=True, help="saved PipelineModel")
    score.add_argument("--results", required=True, help="CSV with labels next to predictions")
    score.add_argument("--summary", required=True, help="text file for the results")
    return parser.parse_args()


def export_sample(spark, args):
    rows = (spark.read.parquet(args.input)
            .select("id", "text")
            .orderBy(rand(args.seed))
            .limit(args.n)
            .collect())
    # utf-8-sig so Excel and Google Sheets show emoji correctly
    with open(args.sheet, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_no", "id", "text", "manual_label"])
        for i, row in enumerate(rows, start=1):
            writer.writerow([i, row["id"], row["text"], ""])
    print(f"Wrote {len(rows)} messages (seed {args.seed}) to {args.sheet}")
    print(f"Fill manual_label with one of: {', '.join(CLASS_NAMES)}, {NONE_LABEL}")


def read_labels(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    problems = []
    for row in rows:
        label = LABEL_ALIASES.get(row["manual_label"].strip().lower())
        if label is None:
            problems.append(f"  row {row['sample_no']}: '{row['manual_label']}'")
        row["manual_label"] = label
    if problems:
        raise SystemExit("Missing or unknown manual labels:\n" + "\n".join(problems))
    return rows


def score(spark, args):
    labelled = read_labels(args.sheet)
    model = PipelineModel.load(args.model_path)

    df = spark.createDataFrame([(r["sample_no"], r["text"]) for r in labelled], ["sample_no", "text"])
    predicted = {r["sample_no"]: r for r in
                 model.transform(df)
                 .select("sample_no", col("prediction").cast("int").alias("prediction"),
                         vector_to_array("probability").alias("probs"))
                 .collect()}

    results = []
    for r in labelled:
        p = predicted[r["sample_no"]]
        results.append({
            "sample_no": r["sample_no"], "id": r["id"], "text": r["text"],
            "manual_label": r["manual_label"],
            "predicted_class": CLASS_NAMES[p["prediction"]],
            "predicted_probability": round(max(p["probs"]), 4),
            "probability": ",".join(f"{x:.4f}" for x in p["probs"]),
        })
    for r in results:
        if r["manual_label"] == NONE_LABEL:
            r["correct"] = "n/a"
        else:
            r["correct"] = "yes" if r["manual_label"] == r["predicted_class"] else "no"

    with open(args.results, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    fits = [r for r in results if r["manual_label"] != NONE_LABEL]
    correct = [r for r in fits if r["correct"] == "yes"]
    none_count = len(results) - len(fits)

    lines = [f"Hand-labelled messages: {len(results)}",
             f"Labelled 'None of these': {none_count}",
             f"Labelled with one of the four classes: {len(fits)}"]
    if fits:
        lines.append(f"Accuracy on those {len(fits)}: {len(correct)}/{len(fits)} = {len(correct) / len(fits):.1%}")

    lines += ["", "Manual label (rows) vs predicted class (columns):",
              "  " + " " * 14 + "".join(f"{n:>10}" for n in CLASS_NAMES)]
    for label in CLASS_NAMES + [NONE_LABEL]:
        counts = [sum(1 for r in results if r["manual_label"] == label and r["predicted_class"] == n)
                  for n in CLASS_NAMES]
        lines.append(f"  {label:<14}" + "".join(f"{c:>10}" for c in counts))

    lines += ["", "Misclassified messages (fit a class, model chose another):"]
    for r in fits:
        if r["correct"] == "no":
            text = " ".join(r["text"].split())[:160]
            lines.append(f"  #{r['sample_no']} manual={r['manual_label']} predicted={r['predicted_class']} "
                         f"(p={r['predicted_probability']:.2f}): {text}")

    summary = "\n".join(lines)
    print(summary)
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(summary + "\n")


def main():
    args = parse_args()
    spark = SparkSession.builder.appName("Evaluate").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    if args.step == "sample":
        export_sample(spark, args)
    else:
        score(spark, args)
    spark.stop()


if __name__ == "__main__":
    main()
