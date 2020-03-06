# yolov3.paddle
An yolo v3 implementation with PaddlePaddle.

python inspect_data.py --data_path ../insects/train/
```
valid Images: 1693 total objects: 10347
objs per image [max, avg, min]: [10, 6.1116361488481985, 4]
Generate image: compare_gtbbox_area_to_image_area.png

objs distribution:
0 Boerner       1595
1 Leconte       2216
2 Linnaeus      818
3 acuminatus    953
4 armandi       1765
5 coleoptera    2091
6 linnaeus      909
```
compare_gtbbox_area_to_image_area.png
![compare_gtbbox_area_to_image_area-train](readme_imgs/compare_gtbbox_area_to_image_area-train.png)

Compare to the whole image, the gtbboxes area ratio are really small, most of them fall in between 0.1%~1%, even the biggest one is just around 2.6%.

python inspect_data.py --data_path ../insects/val/
```
valid Images: 245 total objects: 1856
objs per image [max, avg, min]: [10, 7.575510204081633, 6]
```
