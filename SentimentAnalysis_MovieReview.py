# 패키지 불러오기
import os
import random
import time
import datetime
import torch
import argparse

import pandas as pd
import numpy as np

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences

# 데이터 불러오기
def load_data(args):
    temp = pd.read_csv(args.raw_data, sep="\t")
    temp = temp
    document = temp.document.tolist()
    labels = temp.label.tolist()
    return document, labels

# 스페셜 토큰 붙이기
def add_special_token(document):
    added = ["[CLS]" + str(sentence) + "[SEP]" for sentence in document]
    return added

# 단어 분절화 및 토큰별 아이디 맵핑
def tokenization(document, mode="huggingface"):
    if mode == "huggingface":
        tokenizer = BertTokenizer.from_pretrained(
                'bert-base-multilingual-cased', 
                do_lower_case=False,
                )
        tokenized = [tokenizer.tokenize(sentence) for sentence in document]
        ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized]
        return ids

# 패딩
def padding(ids, args):
    ids = pad_sequences(ids, maxlen=args.max_len, dtype="long", truncating='post', padding='post')
    return ids

# 어텐션 마스킹
def attention_mask(ids):
    masks = []
    for id in ids:
        mask = [float(i>0) for i in id]
        masks.append(mask)
    return masks

# 전처리 종합
def preprocess(args):
    document, labels = load_data(args)
    document = add_special_token(document)
    ids = tokenization(document)
    ids = padding(ids, args)
    masks = attention_mask(ids)
    del document
    return ids, masks, labels

# train, test 데이터 분리
def train_test_data_split(ids, masks, labels):
    train_ids, test_ids, train_labels, test_labels = train_test_split(ids, labels, random_state=42, test_size=0.1)
    train_masks, test_masks, _, _ = train_test_split(masks, ids, random_state=42, test_size=0.1)
    return train_ids, train_masks, train_labels, test_ids, test_masks, test_labels

# pytorch 데이터 로더 생성
def build_dataloader(ids, masks, label, args):
    dataloader = TensorDataset(torch.tensor(ids), torch.tensor(masks), torch.tensor(label))
    dataloader = DataLoader(dataloader, sampler=RandomSampler(dataloader), batch_size=args.batch_size)
    return dataloader

# 모델 구축 
def build_model(args):
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=args.num_labels)
    if torch.cuda.is_available():
         device = torch.device("cpu") ### <--- cpu
         print(f"{torch.cuda.get_device_name(0)} available")
         model = model.cuda()
    else:
         device = torch.device("cpu")
         print("no GPU available")
         model = model
    return model, device

##### test 수행 
def test(test_dataloader, model, device):
    # 테스트 모드 전환
    model.eval()
    # 정확도 초기화
    total_accuracy = 0
    for batch in test_dataloader:
        batch = tuple(index.to(device) for index in batch)
        ids, masks, labels = batch

        with torch.no_grad():
            outputs = model(ids, token_type_ids=None, attention_mask=masks)
        pred = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
        true = [label for label in labels.cpu().numpy()]
        accuracy = accuracy_score(true, pred)
        total_accuracy += accuracy
    avg_accuracy = total_accuracy/len(test_dataloader)
    print(f"test AVG accuracy : {avg_accuracy: .2f}")
    return avg_accuracy

# 학습(train) 수행
def train(train_dataloader, test_dataloader, args):
    model, device = build_model(args)

    # 옵티마이저 정의 
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    
    # learning rate decay
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*args.epochs)
    
    # 시드 고정
    random.seed(args.seed_val)
    np.random.seed(args.seed_val)
    torch.manual_seed(args.seed_val)
    torch.cuda.manual_seed_all(args.seed_val)
    
    # 그레디언트 초기화
    model.zero_grad()
    
    for epoch in range(0, args.epochs):
        # 훈련모드 
        model.train()
        # loss와  정확도 초기화
        total_loss, total_accuracy = 0, 0
        print("-"*30)
        for step, batch in enumerate(train_dataloader):
            if step % 500 == 0 :
                print(f"Epoch : {epoch+1} in {args.epochs} / Step : {step}")
            
            # 배치 선정
            batch = tuple(index.to(device) for index in batch)
            ids, masks, labels, = batch
            # forward
            outputs = model(ids, token_type_ids=None, attention_mask=masks, labels=labels)
            # loss 도출
            loss = outputs.loss
            total_loss += loss.item()

            # 정확도 도출
            pred = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
            true = [label for label in labels.cpu().numpy()]
            accuracy = accuracy_score(true, pred)
            total_accuracy += accuracy

            # 그레디언트 연산
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 파라미터 업데이트
            optimizer.step()
            
            # 러닝레이트 최적화
            scheduler.step()
            
            # 그레디언트 초기화
            model.zero_grad()
            
        # epoch 당 loss 와 정확도 계산
        avg_loss = total_loss / len(train_dataloader)
        avg_accuracy = total_accuracy/len(train_dataloader)
        print(f" {epoch+1} Epoch Average train loss :  {avg_loss}")
        print(f" {epoch+1} Epoch Average train accuracy :  {avg_accuracy}")

        # test 수행
        acc = test(test_dataloader, model, device)
        
        # 모델 저장
        os.makedirs("results", exist_ok=True)
        f = os.path.join("results", f'epoch_{epoch+1}_evalAcc_{acc*100:.0f}.pth')
        torch.save(model.state_dict(), f)
        print('Saved checkpoint:', f)

### 실행함수
def run(args):
    ids, masks, labels = preprocess(args)
    train_ids, train_masks, train_labels, test_ids, test_masks, test_labels = train_test_data_split(ids, masks, labels)
    train_dataloader = build_dataloader(train_ids, train_masks, train_labels, args)
    test_dataloader = build_dataloader(test_ids, test_masks, test_labels, args)
    train(train_dataloader, test_dataloader, args)

# Argument parser 부분
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("-raw_data", default="./data/ratings_train.txt")
    parser.add_argument("-raw_data", default="ratings_test.txt")
    parser.add_argument("-max_len", default=128, type=int)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-num_labels", default=2, type=int)
    parser.add_argument("-epochs", default=4, type=int)
    parser.add_argument("-seed_val", default=42, type=int)

    args = parser.parse_args()
    run(args)