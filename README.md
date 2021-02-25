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

##### Command line instructions (git)
You can also upload existing files from your computer using the instructions below.


Git global setup
git config --global user.name "Cao Hai Nam"
git config --global user.email "namch.hust@gmail.com"

Create a new repository
git clone https://gitlab.com/NamCaoHai/siameser.git
cd siameser
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master

Push an existing folder
cd existing_folder
git init
git remote add origin https://gitlab.com/NamCaoHai/siameser.git
git add .
git commit -m "Initial commit"
git push -u origin master

Push an existing Git repository
cd existing_repo
git remote rename origin old-origin
git remote add origin https://gitlab.com/NamCaoHai/siameser.git
git push -u origin --all
git push -u origin --tags
