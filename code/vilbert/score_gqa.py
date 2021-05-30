import argparse
import json
import tqdm


def evaluate(preds_list, truth_dict):
    score = 0.
    for entry in tqdm.tqdm(preds_list, total=len(preds_list)):
        quesid = entry["questionId"]
        pred = entry["prediction"]
        label = truth_dict[quesid]["answer"]
        if pred in label:
            score += 1. #label[ans]
    return score / len(preds_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_file", default="", type=str,
                        help="")
    parser.add_argument("--truth_file", default="", type=str,
                        help="")
    args = parser.parse_args()


    preds_list = json.load(open(args.preds_file))  # [{"questionId": "201307270", "prediction": "snowboarding"}, ...]
    truth_dict = json.load(open(args.truth_file))  # questionId: semantic, entailed, equivalent, question, imageId, isBalanced, groups, answer, semanticStr, annotations, types, fullAnswer
    
    score = evaluate(preds_list, truth_dict)
    print(100*score)

