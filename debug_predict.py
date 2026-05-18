from web.app import predictor
import glob, json
files = glob.glob('processed_dataset/test/*')
if not files:
    print('No sample files in processed_dataset/test to call predictor; aborting test dump')
else:
    path = files[0]
    print('Using sample', path)
    res = predictor.predict_with_advice(path)
    print('Predict_with_advice returned:', res)
    dbg = 'last_predict_debug.json'
    with open(dbg, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('Wrote', dbg)
