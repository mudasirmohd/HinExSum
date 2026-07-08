#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick demo: summarize a Hindi article.

    python demo.py --model cc.hi.300.bin [--ratio 0.25] [file.txt]

If no file is given, a built-in sample article is used. For a fast smoke
test without downloading big embeddings, pass --toy to train a tiny
Word2Vec on the input itself (quality will be poor; for testing only).
"""

import argparse
import sys

SAMPLE = """भारतीय अंतरिक्ष अनुसंधान संगठन ने अपने महत्वाकांक्षी चंद्र मिशन का सफल प्रक्षेपण किया। यह मिशन चंद्रमा के दक्षिणी ध्रुव पर उतरने का प्रयास करेगा। दक्षिणी ध्रुव पर अब तक कोई भी देश सफलतापूर्वक नहीं उतरा है। वैज्ञानिकों का मानना है कि इस क्षेत्र में पानी की बर्फ मौजूद हो सकती है। पानी की मौजूदगी भविष्य के मानव मिशनों के लिए अत्यंत महत्वपूर्ण है। इसरो के अध्यक्ष ने कहा कि यह मिशन देश के लिए गौरव का क्षण है। प्रक्षेपण के समय हजारों लोग अंतरिक्ष केंद्र में उपस्थित थे। लेकिन इस मिशन की राह आसान नहीं थी। पिछला प्रयास अंतिम चरण में विफल हो गया था। वैज्ञानिकों ने पिछली विफलता से सीखकर कई सुधार किए हैं। इस बार लैंडर के पैरों को मजबूत बनाया गया है। इसके अलावा सॉफ्टवेयर में भी महत्वपूर्ण बदलाव किए गए हैं। यह मिशन लगभग चालीस दिनों में चंद्रमा पर पहुंचेगा। मिशन की सफलता से भारत चौथा देश बन जाएगा जो चंद्रमा पर सॉफ्ट लैंडिंग करेगा। दुनिया भर के वैज्ञानिक इस मिशन पर नजर रखे हुए हैं। इस परियोजना की लागत कई अन्य देशों के मिशनों से काफी कम है। कम लागत में प्रभावी मिशन भारत की पहचान बन गई है। सरकार ने अंतरिक्ष क्षेत्र में निजी कंपनियों को भी आमंत्रित किया है। इससे देश में अंतरिक्ष प्रौद्योगिकी का तेजी से विकास होगा। युवा वैज्ञानिकों के लिए यह क्षेत्र नए अवसर लेकर आया है।"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="UTF-8 Hindi text file")
    ap.add_argument("--model", help="Path to Hindi embeddings")
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--toy", action="store_true",
                    help="Train a throwaway Word2Vec on the input (testing)")
    ap.add_argument("--no-pos", action="store_true")
    args = ap.parse_args()

    text = open(args.file, encoding="utf-8").read() if args.file else SAMPLE

    from hinexsum import HindiSummarizer, load_embeddings
    from hinexsum.embeddings import SemanticModel
    from hinexsum.preprocessing import split_sentences, preprocess_sentence

    if args.toy:
        from gensim.models import Word2Vec
        corpus = [preprocess_sentence(s) for s in split_sentences(text)]
        w2v = Word2Vec(corpus, vector_size=50, window=5, min_count=1,
                       sg=1, epochs=200, seed=42)
        model = SemanticModel(w2v.wv, top_m=3, bv_size=60)
        print("[toy embeddings trained on the input itself]\n",
              file=sys.stderr)
    elif args.model:
        model = load_embeddings(args.model)
    else:
        sys.exit("Provide --model PATH or --toy")

    summ = HindiSummarizer(model, use_pos=not args.no_pos)
    res = summ.summarize(text, ratio=args.ratio)

    print(f"Input: {len(res.sentences)} sentences | "
          f"clusters k={res.diagnostics.get('k')} | "
          f"redundant removed={res.diagnostics.get('removed_redundant')} | "
          f"selected={len(res.selected_indices)}\n")
    print("----- SUMMARY -----")
    print(res.summary)


if __name__ == "__main__":
    main()
