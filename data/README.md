# Download the data

1. Download the compressed dataset

    ```bash 
    wget https://nas.chongminggao.top:4430/easyrl4rec/data.tar.gz
    ```
   or you can manually download it from this website:
   https://rec.ustc.edu.cn/share/a3bdc320-d48e-11ee-8c50-4b1c32c31e9c
   


2. Uncompress the downloaded `data.tar.gz`. The following command will directly extract `data.tar.gz` into the `.data/` directory and merge it with the existing files under `.data/`.

   ```bash
   tar -zxvf data.tar.gz
   ```
   Please note that the decompressed file size is as high as 8.1GB. This is due to the large space occupied by the ground-truth of the user-item interaction matrix. 
   
