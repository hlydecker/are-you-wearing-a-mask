# Data

Data goes here. However, I don't own the data, so I am not going to stick it on github.

Instead, let's go get the data!

Download it from [kaggle](https://www.kaggle.com/datasets/aditya276/face-mask-dataset-yolo-format)

Then, in terminal:

```bash
unzip archive.zip 
```

Next, we will need to make sure the files are ordered in the way that YOLO likes.

```bash
mkdir images/test/labels
mkdir images/train/labels
mkdir images/valid/labels
mkdir images/test/images
mkdir images/train/images
mkdir images/valid/images

mv images/test/*.txt images/test/labels
mv images/train/*.txt images/train/labels
mv images/valid/*.txt images/valid/labels

mv images/test/* images/test/images
mv images/train/* images/train/images
mv images/valid/* images/validimages
```

## Another face mask dataset

[This one looks better than kaggle](https://mvrigkas.github.io/FaceMaskDataset/)
