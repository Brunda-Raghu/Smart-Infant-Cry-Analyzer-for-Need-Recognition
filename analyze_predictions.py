import pandas as pd
from predict import InfantCryPredictor
from pathlib import Path
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

p = Path('processed_dataset')
meta = pd.read_csv(p / 'metadata.csv')

test_rows = meta[meta['split'] == 'test']
print('Test set size:', len(test_rows))

predictor = InfantCryPredictor(model_path='deepinfant.pth')

true = []
pred = []
confidences = []
files = []

for _, row in test_rows.iterrows():
    f = p / 'test' / row['filename']
    try:
        label, conf = predictor.predict(str(f))
        true.append(row['class_code'])
        pred.append(label)
        confidences.append(conf)
        files.append(f.name)
    except Exception as e:
        print('Error predicting', f, e)

print('\nPredicted label distribution:')
print(Counter(pred))

labels = sorted(list(set(true+pred)))
print('\nLabels used:', labels)

# Map labels to indices for confusion matrix
label_to_idx = {l:i for i,l in enumerate(labels)}
true_idx = [label_to_idx[l] for l in true]
pred_idx = [label_to_idx[l] for l in pred]

cm = confusion_matrix(true_idx, pred_idx, labels=list(range(len(labels))))
print('\nConfusion Matrix (rows=true, cols=pred):')
print(cm)

print('\nClassification report:')
print(classification_report(true, pred, labels=labels, zero_division=0))

# Confidence stats
import numpy as np
conf_by_pred = {}
for lbl, c in zip(pred, confidences):
    conf_by_pred.setdefault(lbl, []).append(c)
print('\nAverage confidence by predicted label:')
for k,v in conf_by_pred.items():
    print(f'{k}: {np.mean(v):.3f} (n={len(v)})')

# Save results
out = pd.DataFrame({'file':files, 'true':true, 'pred':pred, 'conf':confidences})
out.to_csv('prediction_results.csv', index=False)
print('\nSaved prediction_results.csv')
