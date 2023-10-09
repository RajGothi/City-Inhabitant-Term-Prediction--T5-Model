import wandb
from tqdm import tqdm
import os
from IPython.display import HTML, display
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, Adafactor, T5Tokenizer
import time
import warnings
warnings.filterwarnings('ignore')

wandb.init(
    project='CS626_NLP_Assignment1',
    entity='account'
)

batch_size = 8

is_normal=True
is_indiviual_character = False

repo_name = "train_t5_new"

def progress(loss, value, max=100):
    return HTML(""" Batch loss :{loss}
      <progress
value='{value}'max='{max}',style='width: 100%'>{value}
      </progress>
             """.format(loss=loss, value=value, max=max))

model_checkpoint = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

# train_df = pd.read_csv(
#     '../phoneme_10_split_2892_train_BPE_city_term_dataset_SEP_output_with_space.csv')


train_df = pd.read_csv("train.csv")

test_data = pd.read_csv("test.csv")

total_train_examples = len(train_df)
print(total_train_examples)
num_of_batches = len(train_df)/batch_size

def remove_sep(x):
   x = x.replace('<SEP>','')
  #  print(x)
   return x

def add_space_bw_character(x):
    string = ''.join(x.split('<SEP>'))
    string = list(string)
    string = '<SEP>'.join(string)
    # print(string)
    return string

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")


valid_data = pd.read_csv(
    'validation.csv')

total_valid_examples = len(valid_data)

if is_normal:
  train_df["city"]=train_df['city'].map(remove_sep)
  valid_data['city']=valid_data['city'].map(remove_sep)
  test_data['city']= test_data['city'].map(remove_sep)
  

if is_indiviual_character:
    train_df["city"]=train_df["city"].map(add_space_bw_character)
    valid_data['city']=valid_data['city'].map(add_space_bw_character)
    test_data['city']=test_data['city'].map(add_space_bw_character)

# train_df.to_csv("final_train_dataset.csv")
# valid_data.to_csv("final_valid_dataset.csv")
# test_data.to_csv("final_test_dataset.csv")
# add_data.to_csv("add_data_dataset.csv")

#concatenate the data
# print(len(train_df))


model = T5ForConditionalGeneration.from_pretrained(
    model_checkpoint, return_dict=True, config="./config_flanT5_small.json")
# moving the model to GPU
model.to(dev)

optimizer = Adafactor(model.parameters(), lr=1e-5,
                      eps=(1e-30, 1e-3),
                      clip_threshold=1.0,
                      decay_rate=-0.8,
                      beta1=None,
                      weight_decay=0.05,
                      relative_step=False,
                      scale_parameter=False,
                      warmup_init=False)

num_of_batches = int(num_of_batches)
num_of_epochs = 100
validation_batch_size = 32
model.train()

loss_per_10_steps = []

for epoch in range(1, num_of_epochs+1):
    print('Running epoch: {}'.format(epoch))
    total_train_correct = 0
    total_train_samples = 0
    running_loss = 0
    
    train_df = train_df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame rows

    # out = display(progress(1, num_of_batches+1), display_id=True)
    for i in range(num_of_batches):
        # print(f"epoch: {epoch}, batch: {i}/{num_of_batches}")
        inputbatch = []
        labelbatch = []
        new_df = train_df[i*batch_size:i*batch_size+batch_size]
        # print(new_df[:5])
        target = []
        for indx, row in new_df.iterrows():
            # print(indx, row["city"])
            labels = row['term']+'</s>'
            # input_ = 'predict demonym: '+row['city']+' <SEP> '+ row['phoneme'] +'</s>'
            input_ = 'predict demonym: '+row['city']+ '</s>'
            target.append(row['term'])
            inputbatch.append(input_)
            labelbatch.append(labels)
        inputbatch = tokenizer.batch_encode_plus(
            inputbatch, padding=True, max_length=512, return_tensors='pt')["input_ids"]
        labelbatch = tokenizer.batch_encode_plus(
            labelbatch, padding=True, max_length=512, return_tensors="pt")["input_ids"]
        inputbatch = inputbatch.to(dev)
        labelbatch = labelbatch.to(dev)

        # clear out the gradients of all Variables
        optimizer.zero_grad()

        # Forward propogation
        outputs = model(input_ids=inputbatch, labels=labelbatch)
        predicted_ids_train = torch.argmax(outputs.logits, dim=-1)
        gen_texts_train = []
        for ids in predicted_ids_train:
            gen_texts_train.append(tokenizer.decode(
                ids).replace("<pad>", "").replace("</s>", ""))
        # predicted_ids = torch.argmax(logits, dim=-1)
            # Calculate accuracy for this batch
        total_train_samples += len(new_df)
        total_train_correct += sum(1 for x,
                                   y in zip(gen_texts_train, target) if x == y)

        loss = outputs.loss
        loss_num = loss.item()
        logits = outputs.logits
        running_loss += loss_num
        if i % 10 == 0:
            loss_per_10_steps.append(loss_num)
        # out.update(progress(loss_num,i, num_of_batches+1)
        # calculating the gradients
        loss.backward()
        optimizer.step()

    running_loss = running_loss/int(total_train_samples)

    print('Epoch: {} , Running loss: {}'.format(epoch, running_loss))
    print('Total Correct: ', total_train_correct)
    print('Total Samples: ', total_train_samples)
    accuracy = total_train_correct / total_train_samples
    print('Accuracy: ', accuracy*100)
    wandb.log({"Training Loss": running_loss, "Training Accuracy": accuracy})

    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    validation_batches = [valid_data[i:i + validation_batch_size]
                          for i in range(0, len(valid_data), validation_batch_size)]

    with torch.no_grad():
        batch_loss = 0
        for batch in validation_batches:
            input_texts = []
            label_texts = []
            target = []
            for i in range(len(batch)):
                labels = batch.iloc[i]['term'] + '</s>'            
                # input_text = 'predict demonym: ' + batch.iloc[i]['city'] + ' <SEP> '+ row['phoneme'] + '</s>'
                input_text = 'predict demonym: ' + batch.iloc[i]['city'] + '</s>'
                input_texts.append(input_text)
                label_texts.append(labels)
                target.append(batch.iloc[i]['term'])

            input_ids = tokenizer.batch_encode_plus(
                input_texts, padding=True, max_length=512, return_tensors='pt')["input_ids"].to(dev)
            label_ids = tokenizer.batch_encode_plus(
                label_texts, padding=True, max_length=512, return_tensors='pt')["input_ids"].to(dev)
            outputs = model(input_ids=input_ids, labels=label_ids)
            loss = outputs.loss
            loss_num = loss.item()
            batch_loss += loss_num

            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            gen_texts = []
            for ids in predicted_ids:
                gen_texts.append(tokenizer.decode(ids).replace(
                    "<pad>", "").replace("</s>", ""))
            # predicted_ids = torch.argmax(logits, dim=-1)
                # Calculate accuracy for this batch
            total_samples += len(batch)
            total_correct += sum(1 for x,
                                 y in zip(gen_texts, target) if x == y)

    batch_loss = batch_loss/int(total_valid_examples)
    print('Epoch: {} ,Validation loss: {}'.format(epoch, batch_loss))


    print('Total Correct: ', total_correct)
    print('Total Samples: ', total_samples)
    accuracy = total_correct / total_samples
    print('Validation Accuracy: ', accuracy*100)
    wandb.log({"Validation Loss": batch_loss,"Validation Accuracy": accuracy})

    # Set the model back to training mode
    model.train()
    path = f"{repo_name}"
    if not os.path.isdir(path):
        os.mkdir(path)
    if (epoch % 10 == 0):
        torch.save(model.state_dict(),
                   f'{path}/flanT5checkpoint' + str(epoch) + '.bin')

print('Training completed')

model = model.to('cpu')

def generate(word):
    input_ids=tokenizer.encode('predict demonym: ',word,return_tensors="pt")
    outputs=model.generate(input_ids)
    gen_text=tokenizer.decode(outputs[0]).replace('<pad>','').replace('</s>','')
    gen_text=gen_text.strip()
    return gen_text


correctcount = 0
totalcount = len(test_data)
prediction = []
# yesno = []

for _, row in tqdm(test_data.iterrows(), total=totalcount, desc="Processing"):
    ans = generate(row[1])
    prediction.append(ans)
    special_character = "\u2581"  # Unicode code point for the character ▁
    val = row[2].replace(special_character, "").strip()
    if ans == val:
        correctcount += 1

accuracy = (correctcount / totalcount) * 100
print("Test accuracy:", accuracy)
