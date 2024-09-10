# Creator: aFun
# Contact Information: afunaaa222@163.com
# Institution: GUANGDONG UNIVERSITY OF TECHNOLOGY
# CreateTime: 2024/5/19 16:54
import json

from nltk.translate.chrf_score import corpus_chrf, sentence_chrf

def getChrF(refs, preds):
    c_chrf_scores = []
    for ref, pred in zip(refs, preds):
        c_chrf_score = corpus_chrf([ref], [pred])
        c_chrf_scores.append(c_chrf_score)
    c_chrf_avgScore = sum(c_chrf_scores) / len(c_chrf_scores)

    s_chrf_scores = []
    for ref,pred in zip(refs,preds):
        s_chrf_score = sentence_chrf(ref,pred)
        s_chrf_scores.append(s_chrf_score)
    s_chrf_avgScore = sum(s_chrf_scores) / len(s_chrf_scores)

    return c_chrf_avgScore, s_chrf_avgScore, c_chrf_scores, s_chrf_scores

if __name__ == '__main__':
    with open("./MMCS_20240519_best.jsonl","r") as reader:
        refs = []
        preds = []
        for line in reader.readlines():
            data = json.loads(line)
            refs.append(data["reference"].split())
            preds.append(data["candidate"].split())
        c_chrf_score, s_chrf_avgScore = getChrF(refs, preds)

        print("c_chrf_score = ",c_chrf_score * 100)
        print("s_chrf_avgScore = ",s_chrf_avgScore * 100)