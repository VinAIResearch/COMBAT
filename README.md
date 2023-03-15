# COMBAT: Alternated Training for Effective Clean-Label Backdoor Attack

This repository contains the code to replicate experiments in our paper **COMBAT: Alternated Training for Effective Clean-Label Backdoor Attack**

# Requirements
Install required Python packages:
```
$ python -m pip install -r requirements.txt
```
# Training trigger generator and surrogate model 
Run command
```
$ python train_generator.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
``` 

where the parameters are as following:
- `dataset`: `cifar10` | `imagenet10` | `celeba`
- `pc`: proportion of the target class data to poison on a 0-to-1 scale
- `noise_rate`: strength/amplitude of the backdoor trigger on a 0-to-1 scale

The trained checkpoint of the generator and surrogate model should be saved at the path `checkpoints\<savingPrefix>_clean\<datasetName>\<datasetName>_<savingPrefix>_clean.pth.tar.`

# Train victim model
Run command
```
$ python train_victim.py --dataset <datasetName> --attack_mode <attackMode> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --load_checkpoint <trainedCheckpoints>
```
`load_checkpoint`: trained generator checkpoint folder name.

The trained checkpoint of the victim model should be saved at the path `checkpoints\<savingPrefix>_clean\<datasetName>\<datasetName>_<savingPrefix>_clean.pth.tar.`
# Evaluate victim model
Run command
```
$ python eval.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
```
# Sample run
```
$ python train_generator.py --dataset cifar10 --pc 0.5 --noise_rate 0.08 --saving_prefix train_generator_n008_pc05
$ python train_victim.py --dataset cifar10 --pc 0.5 --noise_rate 0.08 --saving_prefix train_victim_n008_pc05  --load_checkpoint train_generator_n008_pc05_clean
$ python eval.py --dataset cifar10 --pc 0.5 --noise_rate 0.08 --saving_prefix train_victim_n008_pc05  
```
# Pretrained models
We also provide pretrained checkpoints used in the original paper. The checkpoints could be found [here](https://drive.google.com/drive/folders/1YnHTkeSiOzRlXbjd6OKLs9jXHWSikATQ?usp=sharing) (anonymously). You can download and put them in this repository for evaluating.

# Customized attack configurations
To run other attack configurations (warping-based trigger, input-aware trigger, imperceptible trigger, multiple target labels), follow similar steps mentioned above. For example, to run multiple target labels attack, run the commands:
```
$ python train_generator_multilabel.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
$ python train_victim_multilabel.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --load_checkpoint <trainedCheckpoints>
$ python eval_multilabel.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
```
# Defense experiments
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
