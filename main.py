import argparse
from pathlib import Path

from metrics import evaluate
from predict import load_predictor, predict_file, predict_sentence
from trainer import train
from utils import config_path, project_root


def run_train(args):
    train(config_path(args.config))


def run_evaluate(args):
    result = evaluate(args.pred, args.gold, count_duplicates=args.count_duplicates)
    if args.count_duplicates:
        print(f"TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
        print(f"P={result['precision']:.4f}, R={result['recall']:.4f}, F1={result['f1']:.4f}")
    else:
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall:    {result['recall']:.4f}")
        print(f"F1:        {result['f1']:.4f}")


def run_predict(args):
    model, tokenizer, config = load_predictor(config_path(args.config))
    if args.input:
        output = args.output or project_root() / "results" / "custom_predictions.jsonl"
        predict_file(model, tokenizer, args.input, output, config, args.batch_size)
        print(f"Saved predictions to {output}")
    elif args.text:
        print(predict_sentence(model, tokenizer, args.text, config))
    else:
        while True:
            sentence = input("Sentence (q to quit): ").strip()
            if sentence.lower() == "q":
                break
            print(predict_sentence(model, tokenizer, sentence, config))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--config", default="qlora.json")
    train_parser.set_defaults(handler=run_train)

    evaluate_parser = commands.add_parser("eval")
    evaluate_parser.add_argument("--pred", type=Path, required=True)
    evaluate_parser.add_argument(
        "--gold",
        type=Path,
        default=project_root() / "data" / "bc2gm_test.json",
    )
    evaluate_parser.add_argument("--count-duplicates", action="store_true")
    evaluate_parser.set_defaults(handler=run_evaluate)

    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--config", default="predict.json")
    predict_parser.add_argument("--text")
    predict_parser.add_argument("--input", type=Path)
    predict_parser.add_argument("--output", type=Path)
    predict_parser.add_argument("--batch-size", type=int, default=8)
    predict_parser.set_defaults(handler=run_predict)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
