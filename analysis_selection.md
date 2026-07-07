# Selection-position analysis — XL-Sum Hindi, first 200 test docs (ablation set)

- Model: fastText `cc.hi.300.bin`; same 200 docs and system configs as `results_ablation.md`.
- Budget: 3 sentences. POS on. Positions normalized to [0,1] (0 = article start, 1 = article end) and binned into deciles.
- Companion to `results_ablation.md` / `results_tuned.md`: this shows *where in the article*
  each system draws its sentences from, which is the mechanism behind those ROUGE numbers.

## Fraction of selected sentences by source-position decile

| System | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | mean pos |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours-full | 0.20 | 0.14 | 0.11 | 0.10 | 0.09 | 0.07 | 0.08 | 0.04 | 0.07 | 0.08 | 0.391 |
| Ours-global-topn | 0.20 | 0.12 | 0.12 | 0.11 | 0.09 | 0.08 | 0.07 | 0.05 | 0.07 | 0.08 | 0.394 |
| Ours-position-only | 0.71 | 0.20 | 0.08 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.070 |
| Ours-minus-position | 0.12 | 0.09 | 0.10 | 0.10 | 0.09 | 0.09 | 0.09 | 0.06 | 0.12 | 0.14 | 0.501 |
| Lead-3 | 0.71 | 0.20 | 0.08 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.070 |
| TextRank-3 | 0.15 | 0.10 | 0.11 | 0.10 | 0.09 | 0.12 | 0.10 | 0.08 | 0.09 | 0.06 | 0.441 |
| Random-3 | 0.13 | 0.10 | 0.10 | 0.09 | 0.07 | 0.12 | 0.11 | 0.09 | 0.10 | 0.11 | 0.488 |

D1 = first 10% of the article … D10 = last 10%. Rows sum to ~1.00 across D1–D10.

### How to read this (and why it explains the ROUGE gaps)

- **Ours-position-only ≡ Lead-3** — identical distributions (0.71 / 0.20 / 0.08 in D1–D3,
  mean position **0.070**). Selection is locked to the article's opening. These two also
  posted the top R1-F (0.228) in the ablation. On XL-Sum, where the reference is a
  ~1-sentence lead-style summary, *being at the top of the article is the whole game*.
- **Ours-minus-position is the most back-loaded system of all — mean position 0.501, with
  its single largest bin in D10 (the last 10%).** Dropping position lets the embedding /
  tf-idf / cue features chase lexically "central" sentences, which in news writing sit in
  the body and tail (rosters, quotes, background). This is exactly why it scored *worst*
  (0.182) — it is systematically reading the wrong end of the article.
- **Equal-weight variants (Ours-full 0.391, Ours-global-topn 0.394) land in between.**
  Position is present but diluted to 1/7 of the score, giving only a mild lead lean
  (D1 = 0.20 vs Lead's 0.71). That mild lean is worth just a little R1-F and nowhere near
  enough to match Lead-3 — quantifying the "position gets averaged away" story.
- **Random-3 (0.488) and TextRank-3 (0.441)** are near-uniform, as expected; TextRank's
  centrality gives it a faint lead tilt but nothing decisive.

**Takeaway:** the ROUGE ranking (position-only/Lead > equal-weight > minus-position) is a
direct shadow of *where* each system reads. Beating Lead-3 requires selecting non-lead
sentences that are still in the reference — and on XL-Sum those barely exist.

## Examples — Ours-minus-position vs Lead-3 vs reference

Three documents where dropping the position feature pulls selection furthest from the lead (largest mean selected position). Sentences shown as `[source_index] text`.

### Example 1 — test doc #118 (14 sentences)

**Reference summary:** स्विटज़रलैंड के रोजर फ़ेडरर ने थाइलैंड ओपन टेनिस प्रतियोगिता का पुरुष एकल ख़िताब एक बार फिर जीत लिया है. भारत के लिएंडर पेस ने ऑस्ट्रेलिया के पॉल हैनली के साथ पुरुषों का डबल्स ख़िताब जीता है.

**Lead-3** (positions [0, 1, 2]):
- `[0]` पहली वरीयता प्राप्त फ़ेडरर ने रविवार को फ़ाइनल मुक़ाबले में ब्रिटेन के एंडी मरे को सीधे सेटों में 6-3, 7-5 से हरा दिया.
- `[1]` ब्रिटेन के किशोर खिलाड़ी मरे का यह पहला एटीपी फ़ाइनल था, और उन्होंने चोटी के खिलाड़ी का जम कर मुक़ाबला किया.
- `[2]` मरे के दो डबल फ़ॉल्ट के कारण पहले सेट में नौ मिनट के भीतर ही फ़ेडरर 3-0 से आगे चले गए थे.

**Ours-minus-position** (positions [11, 12, 13]; normalized 11(0.85), 12(0.92), 13(1.00)):
- `[11]` पेस को भी मिली ट्राफ़ी भारत के लिएंडर पेस ने ऑस्ट्रेलिया के पॉल हैनली के साथ पुरुषों का डबल्स ख़िताब जीता है.
- `[12]` पेस-हैनली जोड़ी ने इसराइल जोनाथन एरलिच और एंडी राम को तीन सीधे सेटों में 5-6, 6-1 और6-2 से हराया.
- `[13]` इससे पहले सेमीफ़ाइनल में पेस-हैनली ने भारत के महेश भूपति और अमरीका के जस्टिन गिमेल्सटॉब की जोड़ी को हराया था.

### Example 2 — test doc #12 (13 sentences)

**Reference summary:** सोफ़िया न सुन सकती है न बोल सकती है...और भारत के ग़रीब परिवार की सदस्य होने के नाते उसका भविष्य बहुत उज्जवल नहीं है.

**Lead-3** (positions [0, 1, 2]):
- `[0]` ऐसे किसी देश में जहाँ विकलांगता अपने आपमें ग़रीबी का कारण हो सकती है वहाँ सोफ़िया को काम मिलने की उम्मीद कम ही थी.
- `[1]` और जब काम नहीं मिलता तो यह उम्मीद भी कम ही थी कि वह अपने दहेज के लिए पैसे जुटा पाती जिससे कि उसकी शादी हो जाए.
- `[2]` लेकिन वह 16 साल की ही थी जब उसे केरल में गूंगों और बहरों के लिए खुले स्कूल में भर्ती होने का प्रस्ताव मिला.

**Ours-minus-position** (positions [10, 11, 12]; normalized 10(0.83), 11(0.92), 12(1.00)):
- `[10]` यह संस्था 'ग्रीन बेबी' नाम की कंपनी के लिए काम करती है जो पश्चिमी देशॉ के बच्चों के लिए 'ऑर्गेनिक' कपड़े बेचती है.
- `[11]` सिस्टर बेस्टी कहती हैं, ''यह सामाजिक कार्य इसलिए महत्वपूर्ण है क्योंकि यह लड़कियों को रोज़गार उपलब्ध कराता है और उनको बेहतर ज़िंदगी के अवसर उपलब्ध करा…
- `[12]` ग्रीन बेबी की प्रवक्ता जिल बार्कर का कहना है कि इस तरह की परियोजनाएँ भारत जैसे देशों में बहुत कारगर हो सकती हैं.

### Example 3 — test doc #179 (17 sentences)

**Reference summary:** भारत और बांग्लादेश के ख़िलाफ़ दूसरा टेस्ट शुक्रवार से चटगाँव में शुरू हो रहा है. पहला टेस्ट जीतने के बाद भारतीय टीम दो टेस्ट मैचों की सिरीज़ 2-0 से जीतने के लिए मैदान पर उतरेगी.

**Lead-3** (positions [0, 1, 2]):
- `[0]` इस टेस्ट में भी नज़र होगी मास्टर ब्लास्टर सचिन तेंदुलकर पर जिन्होंने ढाका टेस्ट में अपने जीवन की सर्वश्रेष्ठ पारी खेलते हुए सुनील गावसकर के 34 शतकों क…
- `[1]` सचिन चाहेंगे कि चटगाँव में वे ये रिकॉर्ड तोड़ दे.
- `[2]` इसके साथ-साथ सचिन को टेस्ट क्रिकेट में 10 हज़ार रन पूरे करने के लिए 157 रन और चाहिए.

**Ours-minus-position** (positions [13, 15, 16]; normalized 13(0.81), 15(0.94), 16(1.00)):
- `[13]` इसलिए इस टेस्ट में उन्हें संघर्ष में रखना हमारा पहला लक्ष्य होगा." बशर ने बताया कि उनकी टीम नए गेंद से अच्छा प्रदर्शन नहीं कर पाई है इसलिए वे इस क्षेत…
- `[15]` भारतीय टीम (इनमें से चुनी जाएगी) सौरभ गांगुली (कप्तान), वीरेंदर सहवाग, गौतम गंभीर, राहुल द्रविड़, सचिन तेंदुलकर, वीवीएस लक्ष्मण, दिनेश कार्तिक, इरफ़ान…
- `[16]` बांग्लादेश टीम (इनमें से चुनी जाएगी) हबीबुल बशर (कप्तान), ख़ालिद मसूद, नफ़ीस इक़बाल, जावेद उमर, मोहम्मद अशरफ़ुल, आफ़ताब अहमद, मोहम्मद रफ़ीक़, तापश बैश…

