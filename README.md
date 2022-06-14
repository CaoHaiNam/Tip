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
###### install jupyter notebook on server
```
pip install notebook
sudo snap install jupyter
```
###### access by port
```
jupyter notebook --port <port_name>
```

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


##### push code to github
https://exerror.com/remote-support-for-password-authentication-was-removed-on-august-13-2021-please-use-a-personal-access-token-instead/

##### kill port ubuntu
sudo kill -9 $(sudo lsof -t -i:port)
https://stackoverflow.com/questions/9346211/how-to-kill-a-process-on-a-port-on-ubuntu

