# COMBAT: Alternated Training for Effective Clean-Label Backdoor Attack

COMBAT is a novel mechanism for creating highly effective clean-label attacks using a trigger pattern generator trained alongside a surrogate model. This flexible approach allows for various backdoor trigger types and targets, achieving near-perfect attack success rates and evading all advanced backdoor defenses, as demonstrated through extensive experiments on standard datasets (CIFAR-10, CelebA, ImageNet-10).

Details of the implementation and experimental results can be found in [our paper](https://ojs.aaai.org/index.php/AAAI/article/view/28019). This repository includes:

- Training and evaluation code.
- Defense experiments.
- Pretrained checkpoints.

If you find this repo useful for your research, please consider citing our paper
```
@inproceedings{huynh2024combat,
  title={COMBAT: Alternated Training for Effective Clean-Label Backdoor Attacks},
  author={Huynh, Tran and Nguyen, Dang and Pham, Tung and Tran, Anh},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={3},
  pages={2436--2444},
  year={2024}
}
```

## Requirements
Install required Python packages:
```
$ python -m pip install -r requirements.txt
```
## Training clean model
Run command
```
$ python train_clean_classifier.py --dataset <datasetName> --saving_prefix <cleanModelPrefix>
```

where the parameters are as following:
- `dataset`: name of the dataset used for training (`cifar10` | `imagenet10` | `celeba`)
- `saving_prefix`: prefix for saving the trained clean model checkpoint
  
The trained checkpoint of the clean model should be saved at the path `checkpoints\<cleanModelPrefix>\<datasetName>\<datasetName>_<cleanModelPrefix>.pth.tar.`


## Training trigger generator and surrogate model 
Run command
```
$ python train_generator.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --load_checkpoint_clean <cleanModelPrefix>
``` 

where the parameters are as following:
- `dataset`: name of the dataset used for training (cifar10 | imagenet10 | celeba)
- `pc`: proportion of the target class data to poison on a 0-to-1 scale
- `noise_rate`: strength/amplitude of the backdoor trigger on a 0-to-1 scale
- `saving_prefix`: prefix for saving the trained generator and surrogate model checkpoint
- `load_checkpoint_clean`: prefix of the trained clean model checkpoint

The trained checkpoint of the generator and surrogate model should be saved at the path `checkpoints\<savingPrefix>_clean\<datasetName>\<datasetName>_<savingPrefix>_clean.pth.tar.`

## Train victim model
Run command
```
$ python train_victim.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --load_checkpoint <trainedCheckpoint>
```
`load_checkpoint`: trained generator checkpoint folder name.

The trained checkpoint of the victim model should be saved at the path `checkpoints\<savingPrefix>_clean\<datasetName>\<datasetName>_<savingPrefix>_clean.pth.tar.`
## Evaluate victim model
Run command
```
$ python eval.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --load_checkpoint_clean <cleanModelPrefix> --load_checkpoint <trainedCheckpoint>
```
## Sample run
```
$ python train_clean_classifier.py --dataset cifar10 --saving_prefix classifier_clean
$ python train_generator.py --dataset cifar10 --pc 0.5 --noise_rate 0.08  --saving_prefix train_generator_n008_pc05 --load_checkpoint_clean classifier_clean
$ python train_victim.py --dataset cifar10 --pc 0.5 --noise_rate 0.08 --saving_prefix train_victim_n008_pc05  --load_checkpoint train_generator_n008_pc05_clean
$ python eval.py --dataset cifar10 --pc 0.5 --noise_rate 0.08 --saving_prefix train_victim_n008_pc05 --load_checkpoint_clean classifier_clean --load_checkpoint train_generator_n008_pc05_clean
```
## Pretrained models
We also provide pretrained checkpoints used in the original paper. The checkpoints could be found [here](https://drive.google.com/drive/folders/1YnHTkeSiOzRlXbjd6OKLs9jXHWSikATQ?usp=sharing). You can download and put them in this repository for evaluating.

## Customized attack configurations
To run other attack configurations (warping-based trigger, input-aware trigger, imperceptible trigger, multiple target labels), follow similar steps mentioned above. For example, to run multiple target labels attack, run the commands:
```
$ python train_generator_multilabel.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --load_checkpoint_clean <cleanModelPrefix>
$ python train_victim_multilabel.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --load_checkpoint <trainedCheckpoint>
```
## Defense experiments
We also provide code of defense methods evaluated in the paper inside the folder `defenses`.
- **Fine-pruning**: We have separate code for different datasets due to network architecture differences. Run the command
```
$ cd defenses/fine_pruning
$ python fine-pruning.py --dataset <datasetName> --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --outfile <outfileName>
```
The results will be printed on the screen and written in file `<outfileName>.txt`
- **STRIP**: Run the command
```
$ cd defenses/STRIP
$ python STRIP.py --dataset <datasetName> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
```
The results will be printed on the screen and all entropy values are logged in `results` folder.
- **Neural Cleanse**: Run the command
```
$ cd defenses/neural_cleanse
$ python neural_cleanse.py --dataset <datasetName> --saving_prefix <savingPrefix>
```
The result will be printed on screen and logged in `results` folder.
- **GradCAM**: Run the command
```
$ cd defenses/gradcam
$ python gradcam.py --dataset <datasetName> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
```
The result images will be stored in the `results` folder.
