model_dir = 'models/spelling/convnet/766b63c661ab11e59bdf22000b1a09df/'
model, data, target_one_hot, target = load_model(model_dir)
n_classes = target_one_hot.shape[1]

probs = model.predict_proba(data, verbose=1)
preds = np.argmax(probs, axis=1)
ranks = np.zeros_like(preds)

for i in range(len(preds)):
    inorder = np.argsort(probs[i, :])
    ranks[i] = n_classes - np.where(inorder == target[i])[0]
