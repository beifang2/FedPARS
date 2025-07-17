## Prototype-Aligned Federated Learning for Robust Object Extraction in Heterogeneous Remote Sensing

The overall directory structure is as follows:

```
FedPARS/
 ├── dataset/                   # Dataset directories
 │   ├── data-bhpools/          # Dataset 1
 │   ├── data-inria/            # Dataset 2
 │   └── data-glm/              # Dataset 3
 ├── src/
 │   ├── client/                # Client-side
 │   ├── model/                 # Model architecture
 │   ├── server/                # Server-side
 │   ├── utils/                 # Utility functions
 │   ├── dataloader.py          # DataLoader
 │   ├── loss.py                # Loss functions
 │   ├── main.py                # Main File
 │   └── options.py             # Configuration options
 └── README.md                  # README
```

## Environment Dependencies

The following environment was used during development:

- **Python**: 3.11  
- **PyTorch**: 2.2.2+cu121  
- **CUDA Toolkit**: 12.1
- **NVIDIA Driver**: 572.16  
- **GDAL**: 3.6.2  

## Dataset

The experiments in this project are based on three widely used remote sensing datasets for object extraction tasks:

- [IAIL](https://project.inria.fr/aerialimagelabeling/): High-resolution aerial imagery labeled for building across diverse urban areas.
- [BH-POOLS](http://patreo.dcc.ufmg.br/2020/07/29/bh-pools-watertanks-datasets/): Satellite images focused on detecting swimming pools and water tanks.
- [GLM](https://github.com/zxk688/GVLM): Satellite imagery with annotated landslide regions across multiple areas.

> All datasets are required to be converted into `.tif` format before training. Please ensure that the GDAL library is installed and properly configured for handling GeoTIFF files.

## How to Use

To reproduce our results or adapt FedPARS to your own dataset, please follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/beifang2/FedPARS.git
```

### 2. Navigate to the Source Directory

```bash
cd FedPARS/src
```

### 3. Run the Main Script

```bash
python main.py
```

This will:

- Load dataset(s) from `../dataset/`
- Initialize the federated learning server and clients
- Begin training

## Citation

```
@article{chen2025prototype,
  title={Prototype-Aligned Federated Learning for Robust Object Extraction in Heterogeneous Remote Sensing},
  author={Chen, Guangsheng and Li, Ming and Yuan, Ye and Lin, Moule and Zhang, Lianchong and Li, Chao and Zou, Weitao and Jing, Weipeng and Emam, Mahmoud},
  journal={IEEE Internet of Things Journal},
  year={2025},
  publisher={IEEE}
}
```