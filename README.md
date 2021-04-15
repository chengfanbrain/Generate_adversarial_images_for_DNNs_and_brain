## Generate adversarial images that fool multiple DNNs and human brain

This is an implemention of the idea in paper "Delving into Transferable Adversarial Examples and Black-box Attacks." Yanpei Liu, Xinyun Chen, Chang Liu, Dawn Song. *ICLR*. 2017.

### Requirements

* Python >=3.7
* Pytorch 1.4.0
* torchvision 0.5.0
* CUDA 10.0
* pillow 6.2.0

### How to use
Locate the image folder in the same directory with script `generate_advImg.py` and `libraries`, and run the following command:
```
python generate_advImg.py --purturbation_level 8 --target_category INDEX_IN_1000_IMAGENET_CATEGORIES --img_dir PATH_TO_ORIGINAL_IMAGE_FOLDER
```

### Examples
* The adversarial images that fools multiple DNNs are more likely to show higher-level features of the target category 
![Example1](https://github.com/chengfanbrain/Generate_adversarial_images_for_DNNs_and_brain/blob/master/examples/Exampe1.jpg)

* The effect of purturbation level
![Example1](https://github.com/chengfanbrain/Generate_adversarial_images_for_DNNs_and_brain/blob/master/examples/Example2.jpg)




