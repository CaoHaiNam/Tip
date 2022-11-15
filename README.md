##### Cách sử dụng markdown cho file readme
https://viblo.asia/helps/cach-su-dung-markdown-bxjvZYnwkJZ

##### sử dụng kết hợp colab, drive và github
https://medium.com/@ptpuyen1511/s%E1%BB%AD-d%E1%BB%A5ng-k%E1%BA%BFt-h%E1%BB%A3p-google-colab-github-google-drive-66716e4e62e2

##### tải dữ liệu vào colab
https://silicondales.com/tutorials/g-suite/how-to-wget-files-from-google-drive/

##### install requirements
python3 -m pip install -r requirements.txt

##### Giải thích về overfit và regularization
https://www.datacamp.com/community/tutorials/towards-preventing-overfitting-regularization
https://viblo.asia/p/cac-phuong-phap-tranh-overfitting-gDVK24AmlLj

##### Giải thích về activation function trong mạng nơ-ron
https://aicurious.io/posts/2019-09-23-cac-ham-kich-hoat-activation-function-trong-neural-networks/

##### Thao tác với Git
https://techmaster.vn/posts/35408/huong-dan-day-code-len-github

##### Hướng dẫn cài đặt anaconda cơ bản trên ubuntu
https://www.youtube.com/watch?v=DY0DB_NwEu0&t=440s

##### Truy cập vào ổ đĩa khác từ cmd
http://windows.mercenie.com/windows-8/access-files-and-folders-using-command-prompt/#:~:text=To%20open%20%E2%80%9Ccmd%20prompt%E2%80%9D%20type,d%E2%80%9D%20drive%20of%20your%20computer.

##### Load model dùng tf.keras để load chứ ko dùng keras để tránh lỗi

##### Lưu json file
```
with open(filename, 'w', encoding='utf-8') as f:
  json.dump(x, f, ensure_ascii=False, indent=4)
```
##### Load json file
```
with open(filename, 'r', encoding='utf-8') as f:
    new_x_dict = json.load(f)
```
##### Write txt file
```
outF = open(filename, "w")
for line in "data":
  # write line to output file
    line = ','.join([line[0], str(line[1])])
    outF.write(line)
    outF.write("\n")
outF.close()
```

##### Read txt file
```
with open(filename) as f:
    vnw = []
    for line in f:
        line = line.strip('\n').lower()
        vnw.append(line)
```
##### Read csv file
```
df = pd.read_csv(filename, index_col=False)
```
##### Save csv file
```
df.to_csv(filename, index=False)
```


https://github.com/UKPLab/sentence-transformers/issues/405

##### In số lượng tham số của model pytorch
https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

https://towardsdatascience.com/linear-algebra-explained-in-the-context-of-deep-learning-8fcb8fca1494
https://towardsdatascience.com/probability-and-statistics-explained-in-the-context-of-deep-learning-ed1509b2eb3f

##### Relation between num of samples and num of parameters
https://stats.stackexchange.com/questions/329861/what-happens-when-a-model-is-having-more-parameters-than-training-samples

##### Hypernym, hyponym
Thực ra đây là 2 khái niệm có ý nghĩa gần tương tự nhau. Ví dụ dễ hiểu: chào mào, sẻ, chìa vôi là hyponym của chim. Chim là hypernym của chào mào, sẻ, chìa vôi

##### phân biêt logits và log probability
....

##### jupyter notebook
* install jupyter notebook on server
```
pip install notebook
sudo snap install jupyter
```
* access by port
```
jupyter notebook --port <port_name>
```
* Link fix loi: https://stackoverflow.com/questions/42648610/error-when-executing-jupyter-notebook-no-such-file-or-directory

##### gcloud tpu usage tutorial
https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm
###### debug
https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1
###### colab with tpu
[ML Frameworks: Hugging Face Accelerate w/ Sylvain Gugger](https://www.youtube.com/watch?v=A7lnu-ZsFZs&t=1840s)

```
accelerate config
```

##### command for gcloud tpu
ssh <br>
```
gcloud alpha compute tpus tpu-vm ssh <node_name> --zone europe-west4-a -- -L <port>:localhost:<port>
```
about jupyter notebook
```
```
chạy accelerate thì ko chạy được jax

#### Tất tần tật về transformers
##### Test with transformers
```
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('bert-base-cased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
text = 'this is a test'
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
ids2token = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
```

##### Save model
###### transformers
```
model.save_pretrained(model_dir)
```
###### SentenceTransformer
```
model.save(model_dir)
```

##### push code to github
https://exerror.com/remote-support-for-password-authentication-was-removed-on-august-13-2021-please-use-a-personal-access-token-instead/

##### kill port ubuntu
```
sudo lsof -i tcp:port
```
sẽ hiện ra các process id (PID) đang dùng port này <br>
kill các PID này 
```
sudo kill PID
```
https://stackoverflow.com/questions/9346211/how-to-kill-a-process-on-a-port-on-ubuntu

##### tạo môi trường với virtualenv <br>
virtualenv -p= python-version env-name <br>
Một cách khác   <br>
python3 -m venv ./venv

##### TRC quota expansion
* If you're interested in additional TRC quota, please let us know how you envision using it and how it would accelerate your research. <br>
Answer: <br>
I have lots of experiences to work with server before, and thanks to clear tpu docs, I dont get stuck of using it. In my research, it has supported me a lot. I can train a very large model continuously day by day, which can not be supported by other enviroment such as colab, kaggle... <br>
I have lots of experiences to work with server before, and thanks to clear tpu docs, I do not get stuck of using it. In my research, it has supported me a lot. My research interest is Natural Language Processing, I can train a very large model such as BERT-base model or GPT2-base model continuously day by day, which can not be supported by other environment such as colab, kaggle... With TPU, there is no barriers to me in order to access, update, dominate, train or fine tune SOTA models in NLP, even it is a very big model. <br>

* Detailed feedback about getting started with TRC: <br>
The guidance is really detailed and easy to understand. I have no troubles to get familiar with it. <br>

* What did you do with your TRC quota? (Which models did you train, what datasets did you work with, what code did you write from scratch, etc.)<br>
1. train NLP model such as spell correction base on sequence to sequence model, clustering,...<br>

2. I fine tune GPT2-base model (https://huggingface.co/NlpHUST/gpt2-vietnamese) for Vietnamese comment generation task. 

3. Dataset was crawled and pre-processed and push to huggingface hub (https://huggingface.co/datasets/CaoHaiNam/data_comment)

4. My source code is based on transformers repo (https://github.com/huggingface/transformers) of HuggingFace team for training and fine-tuning task.

* Are you working on any research papers, blog posts, or other publications based on your use of TRC? If so, please let us know, and please include links here if available.
1. Our team have finished a research relevant to financial domain name "MFinBERT: Multilingual Pretrained Language Model For Financial Domain". Paper is accepted in KSE conference, an international forum for presentations, discussions, and exchanges of state-of-the-art research, development, and applications in the field of knowledge and systems engineering. (https://kse2022.tbd.edu.vn/). Because this conference has not happened yet, so I can not share you the paper link. If you care about it, I could send you PDF version.
2. In addition, I am training a siamese-base model for address standardization. This work are skill in processing.

##### Hướng dẫn tạo API bằng python flask
https://www.youtube.com/watch?v=fJz3JTEtJJA

##### Cách kiểm tra version của 1 thư viện bằng pip
```
pip show < lib-name >
```

##### How to download model from kaggle notebook
https://www.kaggle.com/general/65351

##### Linux tool
https://github.com/holianh/Linux_DeepLearning_tools

##### Tất tần tật để test rasa
https://rasa.com/docs/rasa/testing-your-assistant/

##### Python – Import module outside directory
https://www.geeksforgeeks.org/python-import-module-from-different-directory/

##### Đẩy model lên HuggingFace
https://huggingface.co/docs/transformers/v4.16.2/en/model_sharing
https://huggingface.co/transformers/v4.4.2/model_sharing.html

##### Hướng dẫn sử dụng 1.1.1.1 trên ubuntu
https://itrum.org/threads/huong-dan-cai-cloudflare-warp-1-1-1-1-va-su-dung-tren-linux.2207/

##### 20 câu lệnh phổ biến trong linux
https://www.kdnuggets.com/2022/06/20-basic-linux-commands-data-science-beginners.html?fbclid=IwAR0djUjBmlkpdhrBJevH-kshtVGjqkKg0LoYO9_7mjEtSUauKbdUqQcGfFM

##### Cách viết file markdown
https://viblo.asia/helps/cach-su-dung-markdown-bxjvZYnwkJZ

##### How to generate text huggingface
https://huggingface.co/blog/how-to-generate

##### Lỗi tpu đã gặp
1. https://github.com/pytorch/xla/issues/3132
