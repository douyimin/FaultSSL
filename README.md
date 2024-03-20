The pre trained model will be uploaded before March 20th.
# Introduction

**FaultSSL:**
The prevailing methodology in data-driven fault detection leverages synthetic data for training neural networks. However, it grapples with challenges when it comes to generalization in surveys exhibiting complex structures. To enhance the generalization of models trained on limited synthetic datasets to a broader range of real-world data, we introduce FaultSSL, a semi-supervised fault detection framework. This method is based on the classical mean teacher structure, in which its supervised part employs synthetic data and a few 2D labels. The unsupervised component relyies on two meticulously devised proxy tasks, allowing it to incorporate vast unlabeled field data into the training process. The two proxy tasks are PaNning Consistency (PNC) and PaTching Consistency (PTC). PNC emphasizes the feature consistency in overlapping regions between two adjacent views in predicting the model. This allows for the extension of 2D slice labels to the global seismic volume.#xD;PTC emphasizes the spatially consistent nature of faults. It ensures that the predictions for the seismic, whether made on the entire volume or on individual patches, exhibit coherence without any noticeable artifacts at the patch boundaries. While the two proxy tasks serve different objectives, they uniformly contribute to the enhancement of performance. Experiments showcase the exceptional performance of FaultSSL. In surveys where other mainstream methods fail to deliver, we present reliable, continuous, and clear detection results. FaultSSL reveals a promising approach for incorporating large volumes of field data into training and promoting model generalization across broader surveys.

# Quick start
Get test data :  [F3 and Kerry3D](https://drive.google.com/drive/folders/1LEHd2VO9TZTOjrMuAQ7I446OfYDgcdWo?usp=sharing)
    
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    pip install segyio,opencv_python,cigvis
    cp ./download/F3.npy ./FaultSSL/data/
    python prediction.py --input data/F3.npy
![F3](https://github.com/douyimin/FaultSSL/blob/main/results/1.png)
![Kerry](https://github.com/douyimin/FaultSSL/blob/main/results/0.png)
# Cite us
   
    @article{dou2024faultssl,
    title={FaultSSL: Seismic Fault Detection via Semi-supervised learning},
    author={Dou, Yimin and Li, Kewen and Dong, Minghui and Xiao, Yuan},
    journal={Geophysics},
    volume={89},
    number={3},
    pages={1--43},
    year={2024},
    publisher={Society of Exploration Geophysicists}
    }

# Contact us
emindou3015@gmail.com
