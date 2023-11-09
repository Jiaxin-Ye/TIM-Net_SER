import numpy as np
import os
import torch
import argparse
import random
import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from torch_TIMNET import TorchDataset, TorchTIMNET, LabelSmoothingLoss


def set_all_seeds(seed=317):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(x, y):
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    criterion = LabelSmoothingLoss(len(CLASS_LABELS))

    now = datetime.datetime.now()
    now_time = datetime.datetime.strftime(now,'%Y-%m-%d_%H-%M-%S')

    reports = []
    accuracies = []

    kfold = KFold(n_splits=args.split_fold, shuffle=True, random_state=args.random_seed)
    for i, (train, test) in enumerate(kfold.split(x, y)):
        print(f"Training fold: {i}")

        model = TorchTIMNET(
            dropout=args.dropout,
            nb_filters=args.filter_size,
            kernel_size=args.kernel_size,
            dilations=args.dilation_size,
            output_dim=len(CLASS_LABELS)
        )
        params = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {params}")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
        )
        model.to(device)
        train_data = TorchDataset(x[train], y[train])
        valid_data = TorchDataset(x[test], y[test])

        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size
        )
        valid_dataloader = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size
        )
        folder_address = os.path.join(args.model_path, "torch_"+args.data+"_"+str(args.random_seed)+"_"+now_time)
        if not os.path.exists(folder_address):
            os.mkdir(folder_address)
        weight_path=os.path.join(folder_address, str(args.split_fold)+"-fold_weights_best_"+str(i)+".pt")

        for epoch in range(args.epoch):
            train_losses = []
            train_preds = []
            train_labels = []

            model.train()
            for train_x, train_y in train_dataloader:
                optimizer.zero_grad()

                if len(train_y.shape) > 1:
                    train_y = torch.argmax(train_y, dim=1)

                train_x = train_x.to(device)
                train_y = train_y.to(device)
                logits = model(train_x)

                loss = criterion(logits, train_y)
                train_losses.append(loss)
                preds = torch.argmax(logits, axis=1)
                train_preds.extend(preds.tolist())
                train_labels.extend(train_y.tolist())

                loss.backward()
                optimizer.step()

            valid_losses = []
            valid_preds = []
            valid_labels = []

            model.eval()
            for valid_x, valid_y in valid_dataloader:
                with torch.no_grad():
                    if len(valid_y.shape) > 1:
                        valid_y = torch.argmax(valid_y, dim=1)

                    valid_x = valid_x.to(device)
                    valid_y = valid_y.to(device)
                    valid_logits = model(valid_x)

                    valid_loss = criterion(valid_logits, valid_y)
                    valid_losses.append(valid_loss)
                    valid_pred = torch.argmax(valid_logits, axis=1)
                    valid_preds.extend(valid_pred.tolist())
                    valid_labels.extend(valid_y.tolist())

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_valid_loss = sum(valid_losses) / len(valid_losses)

            print(
                f"""Epoch [{epoch+1}], 
                    Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f},
                    Train Accuracy: {accuracy_score(train_labels, train_preds):.2f}, 
                    Valid Accuracy: {accuracy_score(valid_labels, valid_preds):.2f}
                """
            )

        accuracy = accuracy_score(valid_labels, valid_preds)
        accuracies.append(accuracy)
        report = classification_report(
            valid_labels,
            valid_preds,
            target_names=CLASS_LABELS,
            output_dict=True,
        )
        reports.append(report)
        print(
            classification_report(
                valid_labels, valid_preds, target_names=CLASS_LABELS
            )
        )
        print(
            confusion_matrix(
                valid_labels, valid_preds
            )
        )

        model.to(torch.device("cpu"))
        torch.save(model.state_dict(), weight_path)
    
    print(f"Average ACC: {round(sum(accuracies)/args.split_fold, 2)}")


def test(x, y, checkpoint_path):
    criterion = LabelSmoothingLoss(len(CLASS_LABELS))
    accuracies = []
    kfold = KFold(n_splits=args.split_fold, shuffle=True, random_state=args.random_seed)

    for i, (_, test) in enumerate(kfold.split(x, y)):
        print(f"Evaluating fold: {i}")

        weight_path=os.path.join(checkpoint_path, str(args.split_fold)+"-fold_weights_best_"+str(i)+".pt")
        model = TorchTIMNET(
            dropout=args.dropout,
            nb_filters=args.filter_size,
            kernel_size=args.kernel_size,
            dilations=args.dilation_size,
            output_dim=len(CLASS_LABELS)
        )
        model.load_state_dict(torch.load(weight_path))
        model.to(device)
        model.eval()

        valid_data = TorchDataset(x[test], y[test])
        valid_dataloader = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size
        )

        valid_losses = []
        valid_preds = []
        valid_labels = []

        for valid_x, valid_y in valid_dataloader:
            with torch.no_grad():
                if len(valid_y.shape) > 1:
                    valid_y = torch.argmax(valid_y, dim=1)

                valid_x = valid_x.to(device)
                valid_y = valid_y.to(device)
                valid_logits = model(valid_x)

                valid_loss = criterion(valid_logits, valid_y)
                valid_losses.append(valid_loss)
                valid_pred = torch.argmax(valid_logits, axis=1)
                valid_preds.extend(valid_pred.tolist())
                valid_labels.extend(valid_y.tolist())

        avg_valid_loss = sum(valid_losses) / len(valid_losses)

        print(
            f""" 
                Valid Loss: {avg_valid_loss:.4f},
                Valid Accuracy: {accuracy_score(valid_labels, valid_preds):.2f}
            """
        )

        accuracy = accuracy_score(valid_labels, valid_preds)
        accuracies.append(accuracy)
        print(
            classification_report(
                valid_labels, valid_preds, target_names=CLASS_LABELS
            )
        )
    
    print(f"Average ACC: {round(sum(accuracies)/args.split_fold, 2)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--model_path', type=str, default='./Models/')
    parser.add_argument('--result_path', type=str, default='./Results/')
    parser.add_argument('--test_path', type=str, default='./Test_Models/EMODB_46')
    parser.add_argument('--data', type=str, default='EMODB')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.93)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=46)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--filter_size', type=int, default=39)
    parser.add_argument('--dilation_size', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--stack_size', type=int, default=1)
    parser.add_argument('--split_fold', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    if args.data=="IEMOCAP" and args.dilation_size!=10:
        args.dilation_size = 10
        
    set_all_seeds(args.random_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # device = torch.cuda.device(int(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CASIA_CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")#CASIA
    EMODB_CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")#EMODB
    SAVEE_CLASS_LABELS = ("angry","disgust", "fear", "happy", "neutral", "sad", "surprise")#SAVEE
    RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")#rav
    IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad")#iemocap
    EMOVO_CLASS_LABELS = ("angry", "disgust", "fear", "happy","neutral","sad","surprise")#emovo
    CLASS_LABELS_dict = {"CASIA": CASIA_CLASS_LABELS,
                "EMODB": EMODB_CLASS_LABELS,
                "EMOVO": EMOVO_CLASS_LABELS,
                "IEMOCAP": IEMOCAP_CLASS_LABELS,
                "RAVDE": RAVDE_CLASS_LABELS,
                "SAVEE": SAVEE_CLASS_LABELS}
    CLASS_LABELS = CLASS_LABELS_dict[args.data]

    data = np.load("./MFCC/"+args.data+".npy",allow_pickle=True).item()
    x_source = data["x"]
    y_source = data["y"]

    if args.mode=="train":
        train(x_source, y_source)
    elif args.mode=="test":
        test(x_source, y_source, args.test_path)

