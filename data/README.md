# Download the data

1. Download the compressed dataset

    ```bash 
    wget https://nas.chongminggao.top:4430/openrl4rec/environments.tar.gz
    ```
   or you can manually download it from this website:
   https://rec.ustc.edu.cn/share/a0b07110-91c0-11ee-891e-b77696d6db51
   


2. Uncompress the downloaded `environments.tar.gz` and put the files (`data_raw/` folders) to **their corresponding positions** (for each dataset).

   ```bash
   tar -zxvf environments.tar.gz
   ```
   Please note that the decompressed file size is as high as 12GB. This is due to the large space occupied by the ground-truth of the user-item interaction matrix. 
   
