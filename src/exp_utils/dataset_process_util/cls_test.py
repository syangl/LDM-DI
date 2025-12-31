import numpy as np
from sklearn.decomposition import PCA
from transformers import CLIPTokenizer
import matplotlib.pyplot as plt
import clip
import torch
from tqdm import tqdm
import json
import torch.nn as nn
import torch.optim as optim
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from sklearn.model_selection import train_test_split
from exp_utils.load_util.load_data import create_dataset_dataloader
from exp_utils.config_util.attack_config import attack_args, dataset_plot_names, STATISTIC_ROOT_PATH, TRAIN_OUTPUT_DIR, dataset_config_1, HOLD_OUT_OUTPUT_DIR, dataset_config_2


plt.rcParams['font.family'] = 'Times New Roman'

def extract_clip_features(image_tensors, batch_size=32):
    model, _ = clip.load("ViT-B/32", device='cuda')
    model.eval()

    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_tensors), batch_size), desc="Extracting features"):
            batch = image_tensors[i:i + batch_size]
            batch = torch.stack(batch).to('cuda')
            # extract feature
            feats = model.encode_image(batch)
            # normalize
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats.cpu().numpy())

        return np.vstack(features)


def test_distribution_similarity(features1, features2):
    """
    Test Distribution similarity between two datasets.
    """
    X = np.array(features1)
    Y = np.array(features2)

    # Train a classifier to distinguish between the two datasets
    # If the classifier accuracy is close to 0.5, the two datasets are similar (closed to IID)
    X4clf = np.vstack([X, Y])
    y4clf = np.hstack([np.ones(len(X)), np.zeros(len(Y))])
    # Train a binary classifier using 20% of training data.
    # If the classifier can easily distinguish the two datasets through using a small part of data to train,
    # then the two datasets are not IID.
    X4clf_train, X4clf_test, y4clf_train, y4clf_test = train_test_split(X4clf, y4clf, test_size=0.8, random_state=0, shuffle=True)

    X_train_tensor = torch.FloatTensor(X4clf_train)
    y_train_tensor = torch.FloatTensor(y4clf_train)
    X_test_tensor = torch.FloatTensor(X4clf_test)
    y_test_tensor = torch.FloatTensor(y4clf_test)

    # classifier
    input_dim = X4clf_train.shape[1]
    classifier = nn.Sequential(
        nn.Linear(input_dim, 1),
        nn.Sigmoid()
    )

    # check if model is trained
    model_save_path = os.path.join(test_sim_dir, f"{DATASET1_NAME}_{DATASET2_NAME}_iid_classifier.pt")
    if os.path.exists(model_save_path):
        print("Loading trained classifier model...")
        classifier.load_state_dict(torch.load(model_save_path))
    else:
        print("Training classifier model...")
        criterion = nn.BCELoss()
        optimizer = optim.SGD(classifier.parameters(), lr=0.01)

        # start training
        epochs = 1000
        classifier.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = classifier(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        torch.save(classifier.state_dict(), model_save_path)

    # predict
    classifier.eval()
    with torch.no_grad():
        y_pred_proba = classifier(X_test_tensor).squeeze()
        y_pred = (y_pred_proba > 0.5).float()
        clf_acc = (y_pred == y_test_tensor).float().mean().item()

    print(f"Classifier Accuracy: {clf_acc:.4f}")
    print(f"---------------------------------")

    result = {
        "Clf_Acc": clf_acc,
    }

    return result, classifier

if __name__ == "__main__":
    # create_dataset_dataloader needs to load the tokenizer, although we do not use the tokenizer here
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(attack_args.base_model_id, "tokenizer"))

    dataloader1, _ = create_dataset_dataloader(
        TRAIN_OUTPUT_DIR,
        dataset_config_1,
        tokenizer,
        224,  # required size for CLIP ViT-B/32
        32,  
        need_save=False)
    
    dataloader2, _ = create_dataset_dataloader(
        HOLD_OUT_OUTPUT_DIR,
        dataset_config_2,
        tokenizer,
        224,
        32,
        need_save=False)

    # get plot dataset names
    DATASET1_NAME = "Dataset 1"
    DATASET2_NAME = "Dataset 2"
    for key, value in dataset_plot_names.items():
        if dataset_config_1['dataset'].startswith(key):
            DATASET1_NAME = value
            break
    for key, value in dataset_plot_names.items():
        if dataset_config_2['dataset'].startswith(key):
            DATASET2_NAME = value
            break

    dataset1, dataset2 = [], []
    print("Loading images from dataset 1...")
    for batch in tqdm(dataloader1, desc="Loading dataset 1"):
        for img in batch['image']:
            dataset1.append(img)
    print("Loading images from dataset 2...")
    for batch in tqdm(dataloader2, desc="Loading dataset 2"):
        for img in batch['image']:
            dataset2.append(img)

    # extract features
    features1 = extract_clip_features(dataset1)
    features2 = extract_clip_features(dataset2)

    # do similarity test
    test_sim_dir = os.path.join(STATISTIC_ROOT_PATH, "distri_sim_iid")
    os.makedirs(test_sim_dir, exist_ok=True)
    result, classifier = test_distribution_similarity(features1, features2)

    # save
    metrics_save_path = os.path.join(test_sim_dir, f"{DATASET1_NAME}_{DATASET2_NAME}_metrics.json")
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"\nResults and plot saved to: {test_sim_dir}")
