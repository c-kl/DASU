# DASU-Net
## Joint Optimization Learning of High-Low Frequency Filters for Deep Unfolding-Based Depth Map Arbitrary-Scale Super-Resolution

Jialong Zhang, Lijun Zhao*, Jinjing Zhang, Anhong Wang, Huihui Bai

\* Corresponding author


### The architecture of DASU-Net 
[![results](./docs/img/DASU.PNG)](https://github.com/mdcnn/DASU-Net)

### Results:
[![results](./docs/img/NYU.PNG)](https://github.com/mdcnn/DASU-Net)

### Dependencies
- python3.7+
- pytorch1.9+
- torchvision
- [Nvidia Apex](https://github.com/NVIDIA/apex) (python-only build is ok.)

### Datasets
We follow [Tang et al.](https://github.com/ashawkey/jiif) and use the same datasets. Please refer to [here](https://github.com/ashawkey/jiif/blob/main/data/prepare_data.md) to download the preprocessed datasets and extract them into `./data/` folder.

### Pretrained Models
The pretrained models is placed in `workspace/checkpoints` folder.

### Train
```
python main.py
```
### Test

Please execute `bash test.sh` on the terminal for RMSE values and visualization. For PSNR and SSIM metrics, please look `psnr_ssim.py`.

## Ackownledgements
This code is built based on [GeoDSR](https://github.com/nana01219/GeoDSR). We thank the authors for sharing the codes.

