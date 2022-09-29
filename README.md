# COMBAT: Alternated Training for Near-Perfect Clean-Label Backdoor Attack

This repository contains the code to replicate experiments in our paper **COMBAT: Alternated Training for Near-Perfect Clean-Label Backdoor Attack**

# Requirements
- Install required python packages:
```
$ python -m pip install -r requirements.txt
```
- Download and re-organize GTSRB dataset from its official website:
```
$ bash gtsrb_download.sh
```
# Training trigger generator and surrogate model 
Run command
```
$ python train_generator.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
``` 

where the parameters are as following:
- `dataset`: `cifar10` | `gtsrb` | `celeba`
- `pc`: proportion of the target class data to poison on a 0-to-1 scale
- `noise_rate`: strength/amplitude of the backdoor trigger on a 0-to-1 scale

The trained checkpoints should be saved at the path `checkpoints\<savingPrefix>_clean\<datasetName>\<datasetName>_<savingPrefix>_clean.pth.tar.`

# Train victim model
Run command
```
$ python train_victim.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
```
The trained checkpoints should be saved at the path `checkpoints\<savingPrefix>_clean\<datasetName>\<datasetName>_<savingPrefix>_clean.pth.tar.`
# Evaluate victim model
Run command
```
$ python eval.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
```
# Customized attack configurations
To run other attack configurations (warping-based trigger, input-aware trigger, imperceptible trigger, multiple target labels), follow similar steps mentioned above. For example, to run multiple target labels attack, run the commands:
```
$ python train_generator_multilabel.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
$ python train_victim_multilabel.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
$ python eval_multilabel.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
```
# Defense experiments
We also provide code of defense methods evaluated in the paper inside the folder `defenses`.
- **Fine-pruning**: We have separate code for different datasets due to network architecture differences. Run the command
```
$ cd defenses/fine_pruning
$ python fine-pruning-cifar10-gtsrb.py --dataset cifar10 --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --outfile <outfileName>
$ python fine-pruning-celeba.py --dataset celeba --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --outfile <outfileName>
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
$ python neural_cleanse.py --dataset <datasetName> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
```
The result will be printed on screen and logged in `results` folder.
- **GradCAM**: Run the command
```
$ cd defenses/gradcam
$ python gradcam.py --dataset <datasetName> --noise_rate <triggerStrength> --saving_prefix <savingPrefix>
```
The result images will be stored in the `results` folder.
