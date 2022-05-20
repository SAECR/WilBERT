# %%
import transformers
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import random
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
rcParams['figure.figsize'] = 12, 8
from pycm import ConfusionMatrix
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import scipy.stats as st

EPOCHS = 10
lr = 5e-5
BATCH_SIZE = 16
test_size = 0.2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


df = pd.read_excel(r"Input_Comments.xlsx")
df['sentiment'] = df['sentiment'].astype(int)
class_names = ['negative', 'neutral', 'positive']

from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
model = RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base", return_dict=False)
tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")

max_len = 100

class ECRReports(Dataset):
    def __init__(self, reports, targets, tokenizer, max_len):
        self.reports = reports
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.reports)
    def __getitem__(self, item):
        report = str(self.reports[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
        report,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding = 'max_length',
        return_attention_mask=True,
        return_tensors='pt',
         )
        return {
            'report_text': report,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
               }
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ECRReports(
      reports=df.masked_comments.to_numpy(),
      targets=df.sentiment.to_numpy(),
      tokenizer=tokenizer,
      max_len=max_len,
      )
    return DataLoader(
      ds,
      batch_size=batch_size,
      num_workers=0
      )

def sample(seed_list):
    cm_pred = []
    cm_act = []
    results = []
    for i in seed_list:
        class SentimentClassifier(nn.Module):
            def __init__(self, n_classes):
                super(SentimentClassifier, self).__init__()
                self.robbert = RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base", return_dict=False)
                self.drop = nn.Dropout(p=0.1)
                self.out = nn.Linear(self.robbert.config.hidden_size, n_classes)
            def forward(self, input_ids, attention_mask):
                _, pooled_output = self.robbert(
                input_ids=input_ids,
                attention_mask=attention_mask
                )
                output = self.drop(pooled_output)
                return self.out(output)
        robbert = SentimentClassifier(len(class_names))
        robbert = robbert.to(device)
        df_train, df_test = train_test_split(
                          df,
                          test_size=0.2,
                          random_state=i
                          )
        df_val, df_test = train_test_split(
                          df_test,
                          test_size=0.5,
                          random_state=i
                          )                  
        train_data_loader = create_data_loader(df_train, tokenizer, max_len, BATCH_SIZE)
        val_data_loader = create_data_loader(df_val, tokenizer, max_len, BATCH_SIZE)
        test_data_loader = create_data_loader(df_test, tokenizer, max_len, BATCH_SIZE)
        data = next(iter(train_data_loader))
        data.keys()
        model = robbert
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        total_steps = len(train_data_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss().to(device)
        def train_epoch(
            model,
            data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            n_examples
        ):
            model = model.train()
            losses = []
            correct_predictions = 0
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            return correct_predictions.double() / n_examples, np.mean(losses)

        def eval_model(model, data_loader, loss_fn, device, n_examples):
            model = model.eval()
            losses = []
            correct_predictions = 0
            with torch.no_grad():
                for d in data_loader:
                    input_ids = d["input_ids"].to(device)
                    attention_mask = d["attention_mask"].to(device)
                    targets = d["targets"].to(device)
                    outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                    )
                    _, preds = torch.max(outputs, dim=1)
                    loss = loss_fn(outputs, targets)
                    correct_predictions += torch.sum(preds == targets)
                    losses.append(loss.item())
            return correct_predictions.double() / n_examples, np.mean(losses)

        def get_predictions(model, data_loader):
            model = model.eval()
            
            report_texts = []
            predictions = []
            prediction_probs = []
            real_values = []

            with torch.no_grad():
                for d in data_loader:

                    texts = d["report_text"]
                    input_ids = d["input_ids"].to(device)
                    attention_mask = d["attention_mask"].to(device)
                    targets = d["targets"].to(device)

                    outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                    )
                    _, preds = torch.max(outputs, dim=1)

                    probs = torch.nn.functional.softmax(outputs, dim=1)

                    report_texts.extend(texts)
                    predictions.extend(preds)
                    prediction_probs.extend(probs)
                    real_values.extend(targets)
                predictions = torch.stack(predictions).cpu()
                prediction_probs = torch.stack(prediction_probs).cpu()
                real_values = torch.stack(real_values).cpu()
                return report_texts, predictions, prediction_probs, real_values
        history = defaultdict(list)
        best_accuracy = 0
        for epoch in range(EPOCHS):
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)
            train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
            )
            print(f'Train loss {train_loss} accuracy {train_acc}')
            val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
            )
            print(f'Val   loss {val_loss} accuracy {val_acc}')
            print()
            ta = float(train_acc.cpu().numpy())
            tl = float(train_loss)
            va = float(val_acc.cpu().numpy())
            vl = float(val_loss)
            history['train_acc'].append(float(ta))
            history['train_loss'].append(float(tl))
            history['val_acc'].append(float(va))
            history['val_loss'].append(float(vl))
            if val_acc > best_accuracy:
                torch.save(model.state_dict(), 'Robbert_best_model_state.bin')
                best_accuracy = val_acc
        test_acc, _ = eval_model(
            model ,
            test_data_loader,
            loss_fn,
            device,
            len(df_test)
        )
        acc = test_acc.item()
        results.append(acc)
        plt.figure(0)
        plt.plot(history['train_acc'],  color = 'orange')
        plt.plot(history['val_acc'],  color = 'blue' )
        plt.title('Training history RobBERT')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        epochs = list(range(EPOCHS))
        plt.xticks(epochs)
        plt.ylim([0, 1]);
        plt.savefig(f'./Output/R_e{EPOCHS}b{BATCH_SIZE}lr{lr}s{i}ts{test_size}acc{acc}.png')

        plt.figure(1)
        print(history['train_loss'])
        print(history['val_loss'])
        plt.plot(history['train_loss'],  color = 'orange')
        plt.plot(history['val_loss'],  color = 'blue' )
        plt.title('Training loss RobBERT')
        plt.ylabel('Cross Entropy loss')
        plt.xlabel('Epoch')
        epochs = list(range(EPOCHS))
        plt.xticks(epochs)
        plt.ylim([0, 1]);
        plt.savefig(f'./Output/R_loss_e{EPOCHS}b{BATCH_SIZE}lr{lr}s{i}ts{test_size}acc{acc}.png')

        y_report_texts, y_pred, y_pred_probs, y_test = get_predictions(
            model,
            test_data_loader
        )
        print(classification_report(y_test, y_pred, target_names=class_names))
        plt.figure(2+i)
        def show_confusion_matrix(confusion_matrix):
            hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
            hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
            hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
            plt.ylabel('True sentiment')
            plt.xlabel('Predicted sentiment')
            plt.title('Confusion matrix RobBERT')
            plt.subplots_adjust(bottom=0.05);
        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        show_confusion_matrix(df_cm)
        plt.savefig(f'./Output/R_CFM_e{EPOCHS}b{BATCH_SIZE}lr{lr}s{i}ts{test_size}acc{acc}.png', bbox_inches = 'tight')

        pred_test_df = pd.DataFrame(y_report_texts)
        pred_test_df['preds'] = y_pred.tolist()
        pred_test_df['actual'] = y_test.tolist()
        pred_test_df.to_excel(f'./Output/R_e{EPOCHS}b{BATCH_SIZE}lr{lr}s{i}ts{test_size}acc{acc}.xlsx')

        cm_pred.extend(y_pred.tolist())
        cm_act.extend(y_test.tolist())
    
    cm_pred = [int(i) for i in cm_pred]
    cm_act = [int(i) for i in cm_act]
    print(cm_pred)
    print(cm_act)
    print(classification_report(cm_act, cm_pred, target_names = class_names))
    plt.figure(12)
    def show_confusion_matrix(confusion_matrix):
        hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True sentiment')
        plt.xlabel('Predicted sentiment')
        plt.title('Confusion matrix RobBERT')
        plt.subplots_adjust(bottom=0.05);
    cm = confusion_matrix(cm_act, cm_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)
    plt.savefig(f'./Output/R_CFM_e{EPOCHS}b{BATCH_SIZE}lr{lr}s{10}ts{test_size}acc{sum(results)/len(results)}.png',bbox_inches='tight')


    print(results)
    print(f'Mean accuracy : {sum(results) / len(results)}')
    confidence_interval = st.t.interval(0.95, len(results)-1, loc=np.mean(results), scale=st.sem(results))
    print(f'Confidence interval : {confidence_interval}') 
seed_list = [0,1,2,3,4,5,6,7,8,9]
sample(seed_list)
