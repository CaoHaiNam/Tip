##### chạy ngrok với domain
```
ngrok http port --domain domain_name <br>
```
Ex: ngrok http 3000 --domain wondrous-centrally-alpaca.ngrok-free.app


##### docker sample
```
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

#Install libs
RUN apt-get update && apt-get install -y \
            g++ \
            build-essential \
            cmake \         
            pkg-config \
            python3-dev \
            python3-pip \
            nano \
            libgl1-mesa-glx

# Language and timezone
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Ho_Chi_Minh
RUN apt-get install -y tzdata && echo $TZ > /etc/timezone && dpkg-reconfigure -f noninteractive tzdata

# Install packages
RUN pip3 install --upgrade pip
RUN pip3 install flask flask_cors flask_restplus Werkzeug==0.16.0 gunicorn eventlet pyjwt pymongo
RUN pip3 install opencv-python opencv-contrib-python
RUN pip3 install tensorflow-serving-api tensorflow==2.3.2
RUN pip3 install fuzzywuzzy python-Levenshtein nltk==3.6.1 scikit-learn==0.24.0 scikit-image unidecode
RUN apt-get install -y libzbar0
RUN pip3 install python-logstash mrz pyzbar dbr==9.0 imutils
RUN pip3 install markupsafe==2.0.1 flask==1.1.4 Werkzeug==0.16.0 mrz-scanner-sdk
RUN pip3 install mediapipe

# Add
ADD . /api/
WORKDIR /api/
RUN chmod 777 run_production_server.sh
RUN mkdir /public
RUN mkdir -p static
RUN ln -s /api/static/ /public/

CMD ./run_production_server.sh
```

##### scp file from local to tpu machine
```
gcloud alpha compute tpus tpu-vm scp ~/my-file my-tpu: 
```
Ex: my-tpu = namch_hust1_gmail_com@node-4
```
gcloud alpha compute tpus tpu-vm scp faq_v1_db_add_chatGPT_data.json namch_hust1_gmail_com@node-4:/home/namch_hust1_gmail_com
```

##### chạy fastapi và flask
**fastapi**
```
python3 -m uvicorn app_fastapi:app --reload --host localhost --port 3000 # tạo file app_fastapi.py
```
**flask**
```
python3 -m flask --app app_flask run --port 3000 --host localhost --reload # tạo file app_flask.py
```

##### Compiled và interpreted language programing
https://howkteam.vn/course/goc-lap-trinh-vien/compiled-va-interpreted-la-gi-uu-diem-va-nhuoc-diem-3927

##### cách cài đặt vncorenlp
1. cài đặt java
   ```
   sudo apt install --reinstall openjdk-8-jre-headless
   ```
2. cài biến môi trường
   ```
   export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
   export JVM_PATH="/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/libjvm.so"
   ```
3. cài thư viện
   ```
   pip install py_vncorenlp
   ```
4. tạo folder để lưu model
   ```
   mkdir vncorenlp
   ```
5. download model
   ```
   import py_vncorenlp
   py_vncorenlp.download_model(save_dir='vncorenlp')
   ```
   
6. test segmentation
   ```
   import py_vncorenlp
   rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='vncorenlp/')
   text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
   output = rdrsegmenter.word_segment(text)
   print(output)
   ```

##### tạo và clone folder (model, dataset) trên huggingface
```
huggingface-cli repo create idea-generation-dataset_v1-0
git clone https://huggingface.co/CaoHaiNam/idea-generation-dataset_v1-0
```

##### setup device to run model
```
export CUDA_VISIBLE_DEVICES=""
```

##### chạy lệnh để active gcloud-tpu
```
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
```

##### check folder size
```
ls -l --block-size=M
```

##### lệnh để show hidden file trong khi chay jupyterlab server
```
jupyter lab --port 8003 --ContentsManager.allow_hidden True
```

##### cách thông nhiều port từ server về local qua giao thức ssh
```
ssh root@202.134.19.49 -L 8081:localhost:8081 -L 9870:localhost:9870 -L 8080:localhost:8080
```

##### Chạy lệnh này để clone git trên server, trong trường hợp clone bị lỗi
```
git config --global http.sslverify "false"
```

##### ssh to gcloud instance
```
ssh <YOUR_USERNAME>@<YOUR_TPU_INSTANCE_PUBLIC_IP> -i <YOUR_PRIVATE_KEY_PATH>
```

##### tạo môi trường ảo trên mac os 
```
python3 -m venv myenv
```

##### push code lên nhiều nhánh cùng 1 lúc.
https://viblo.asia/p/git-nang-cao-git-cherry-pick-RQqKLQ9pZ7z

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
* Link fix lỗi: https://stackoverflow.com/questions/42648610/error-when-executing-jupyter-notebook-no-such-file-or-directory

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

5. Currently, I am working with diffusion model for image generation task. In addition, I am training a Vietnamese spell correction model based on sequence to sequence architecture which is available in huggingface library such as BART (https://huggingface.co/docs/transformers/model_doc/bart) or T5 (https://huggingface.co/docs/transformers/model_doc/t5). 

6. training a Vietnamese Summarization Model for the community. I start with vinai/bartpho-syllable, a pre-trained Sequence-to-Sequence  Model based on Bart for Vietnamese. Data for fine-tuning the summary task is available here: https://huggingface.co/datasets/CaoHaiNam/summarization_wikilingua_vi 

7. training a Roberta model for the Vietnamese community. Data for training is available here: imthanhlv/binhvq_dedup

*<18-05-2023>*

8. continue training a Vietnamese Summarization Model for the community. I start with vinai/bartpho-syllable, a pre-trained Sequence-to-Sequence Model based on Bart for Vietnamese. Data for fine-tuning the summary task is available here: https://huggingface.co/datasets/CaoHaiNam/summarization_wikilingua_vi

9. training a distil-bert model for the Vietnamese community. Data for training is available here: imthanhlv/binhvq_dedup

* Are you working on any research papers, blog posts, or other publications based on your use of TRC? If so, please let us know, and please include links here if available.
1. Our team have finished a research relevant to financial domain name "MFinBERT: Multilingual Pretrained Language Model For Financial Domain". Paper is accepted in KSE conference, an international forum for presentations, discussions, and exchanges of state-of-the-art research, development, and applications in the field of knowledge and systems engineering. (https://kse2022.tbd.edu.vn/). Because this conference has not happened yet, so I can not share you the paper link. If you care about it, I could send you PDF version.
2. In addition, I am training a siamese-base model for address standardization. This work are skill in processing.
3. Our publication now is available here: https://ieeexplore.ieee.org/document/9953749, a pre-trained language model for financial domain.
4. At the moment, our team are carrying out image generation task based on diffusion model (https://github.com/huggingface/diffusers), particularly, I am working in a finance company.  For each advertisement campaign (new product release, promotion on holiday), we would like to send interesting image along with description. So input of our model is a description, and output is a image corresponding to text. 
5. Currently, I am improving the token classification module in our paper: https://arxiv.org/abs/2210.14607. This model aims to classify whether a term is an occupational skill or not. I just use a simple NN for this purpose before and now I would like to use Bert-based model for that.

*<18-05-2023>*

6. improving module synonym prediction of our paper: https://link.springer.com/chapter/10.1007/978-3-031-08530-7_29

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

##### rasa
###### Tất tần tật để test rasa
https://rasa.com/docs/rasa/testing-your-assistant/
###### load và test rasa model
```python3
from rasa.core.agent import Agent
import asyncio
agent = Agent.load(model_path)
message = "something"
result = agent.parse_message(message)
asyncio.run(result)
```
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

##### Test code
Flake8

##### DE
* polar (https://pola-rs.github.io/polars-book/user-guide/introduction.html?fbclid=IwAR2yIqZso6CyygxgayQZQyrQ-BZpvSZMY_CXfkwPm7CDs9xyibX-q9_9XaE) 
* pandasparalell
* kafka
1. https://viblo.asia/p/002-apache-kafka-topic-partition-offset-va-broker-m68Z0eEMlkG

##### Kaggle Notebook for Data Science
1. Clustering
* https://www.kaggle.com/code/marcinrutecki/clustering-methods-comprehensive-study/notebook

2. Data Analysis

##### KL và cross entropy
* Dựa vào hàm loss là thấy sự khác nhau và tại sao trong bài toán classification chỉ cần dùng cross entropy là đc rồi. https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
* Forward and Reverse KL: An explaination <br>
https://dibyaghosh.com/blog/probability/kldivergence.html <br>
https://notesonai.com/KL+Divergence


##### Devops doc
* https://techmaster.vn/posts/37441/cam-nang-cac-tap-lenh-linux-ma-ban-hay-dung?fbclid=IwAR3b1rS0pwnBEZM5A6kGIFsVAYpJEPPdV2OsyfzrbSAlUQgYRh-aN0XjCSE

##### gcloud 
https://cloud.google.com/sdk/auth_success

##### Hướng dẫn lưu cache của huggingface dataset trên máy TPU 
1. Tạo 1 folder, ví dụ tên là cache_datasets 
```
mkdir /home/namch_hust1_gmail_com/cache_datasets
```
2. Đây là folder chứa cache. Mount folder này vào trong ram, và truyền vào size của folder, ví dụ trong trường hợp này là 10MB.
```
sudo mount -t tmpfs -o size=10M,mode=0755 tmpfs /home/namch_hust1_gmail_com/cache_datasets
```
3. Chuyển từ quyền root sang quyền user (namch_hust1_gmail_com) cho folder này 
```
sudo chown namch_hust1_gmail_com cache_datasets 
```
Đã xong. Bây giờ, load datasets, truyền giá trị cache_dir chính là folder vừa tạo 
```python 
dataset = datasets.load_dataset('CaoHaiNam/sonnv_dataset_idea_1k', use_auth_token=True, cache_dir='/home/namch_hust1_gmail_com/cache_datasets/datasets')
```
##### Check folder size
Dùng lệnh ncdu 

##### Triển khai service bằng systemd
* https://www.shubhamdipt.com/blog/how-to-create-a-systemd-service-in-linux/
* https://www.digitalocean.com/community/tutorials/how-to-use-systemctl-to-manage-systemd-services-and-units
* https://www.digitalocean.com/community/tutorials/how-to-deploy-node-js-applications-using-systemd-and-nginx
* https://www.tecmint.com/list-all-running-services-under-systemd-in-linux/

##### Install lfs 
```
!curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
!sudo apt-get install git-lfs
!git lfs install
```
