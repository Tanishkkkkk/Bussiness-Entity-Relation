import re

with open('code/business_entity_resolution/src/train_model.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_func = [
    'def train_lgbm(X_train, y_train, X_val, y_val):\n',
    '    print("\\nTraining LightGBM (v2 - enhanced) ...")\n',
    '    pos_w = (y_train == 0).sum() / (y_train == 1).sum()  # balance\n',
    '    params = {\n',
    '        "objective":        "binary",\n',
    '        "metric":           "binary_logloss",\n',
    '        "learning_rate":    0.03,\n',
    '        "num_leaves":       127,\n',
    '        "max_depth":        8,\n',
    '        "min_child_samples": 30,\n',
    '        "n_estimators":     1000,\n',
    '        "scale_pos_weight": pos_w,\n',
    '        "subsample":        0.8,\n',
    '        "colsample_bytree": 0.8,\n',
    '        "reg_alpha":        0.1,\n',
    '        "reg_lambda":       0.1,\n',
    '        "min_split_gain":   0.01,\n',
    '        "random_state":     SEED,\n',
    '        "n_jobs":           -1,\n',
    '        "verbose":          -1,\n',
    '    }\n',
    '    model = lgb.LGBMClassifier(**params)\n',
    '    t0 = time.time()\n',
    '    model.fit(\n',
    '        X_train, y_train,\n',
    '        eval_set=[(X_val, y_val)],\n',
    '        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(100)],\n',
    '    )\n',
    '    print(f"  Trained in {time.time()-t0:.1f}s  (best iter: {model.best_iteration_})")\n',
    '    return model\n',
]

start = None
end = None
for i, l in enumerate(lines):
    if l.strip().startswith('def train_lgbm('):
        start = i
    if start is not None and i > start and l.strip().startswith('def ') and i > start + 2:
        end = i
        break

if start is None:
    print('ERROR: could not find train_lgbm')
else:
    if end is None:
        end = start + 30
    lines[start:end] = new_func + ['\n', '\n']
    with open('code/business_entity_resolution/src/train_model.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'SUCCESS: replaced lines {start+1} to {end}')
