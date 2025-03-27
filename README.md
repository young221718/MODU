# MODU (Model of Object Detection for Universe)

MODU is a comprehensive toolkit for training and evaluating powerful object detection models. This project supports the design, training, and testing of efficient and scalable object detection models.

## Installation

To use MODU, install the required packages with the following command:

```bash
pip install -r requirements.txt
```

## Training Instructions

MODU supports various training methods. Here are some examples:

### Multi-GPU Training

Use the following command to train using multiple GPUs:

```bash
torchrun --nproc_per_node=2 tools/train.py -c /home/prml/StudentsWork/Chanyoung/workspace/detection/detr/graduate_project/configs/rtdetr_pos_attn/r50_coco_72epc_pos_cond.yml
```

### Specific GPU Training

To train on a specific GPU, execute the command below:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c /home/work/StudentsWork/Chanyoung/workspace/detection/detr/graduate_project/configs/rtdetr/convnext_ag_256dim_coco_48epc_pos.yml
```

### Single GPU Training

For single GPU (GPU 0) training, run the following command:

```bash
python tools/train.py -c /home/work/StudentsWork/Chanyoung/workspace/detection/detr/graduate_project/configs/rtdetr/convnext_ag_256dim_coco_tuning.yml -t /home/work/StudentsWork/Chanyoung/workspace/_experiments/graduate/rtdetr_convnext_ag_256dim_coco_241017/best.pth
```

## Additional Options

- **-r**: Specify the `.pt` file path to resume training from that checkpoint.
- **-t**: Use this option for fine-tuning.
- **--test-only**: Use this option for testing only.
- **--test-only --flops**: This option allows you to check the GFLOPs and the number of parameters.
- **--test-only --cmp**: The results will be saved as a JSON file.

---

## Contributing

If you wish to contribute to this project, please fork the repository, make your changes, and submit a pull request. For any inquiries, feel free to open an issue.

## License

MODU is distributed under the [License Type]. See the [LICENSE](./LICENSE) file for more details.
