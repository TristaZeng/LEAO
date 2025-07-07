<h1> LEAO </h1>

This is the source codes and instructions for <b>LEAO</b>, a latent-space-enhanced computational adaptive optics method exploiting the physical priors encoded in the multi-view measurements from light-field microscopy. This package includes the Python implementation of LEAO aberration estimation, and MATLAB implementation of volume reconstruction. You should be able to download our code and set up the environment within five minutes.

<h2> Content </h2>

<ul>
  <li><a href="#File structure">1. File structure</a></li>
  <li><a href="#Environment">2. Environment</a></li>
  <li><a href="#Aberration estimation with LEAO">3. Aberration estimation with LEAO</a></li>
  <li><a href="#Volume reconstruction">4. Volume reconstruction</a></li>
  <li><a href="#Updates">5. Updates</a></li>
</ul>

<hr>

<h2 id="File structure">1. File structure</h2>

- <code>./aberration_estimation</code> includes the Python codes for LEAO training and testing

- <code>./demo_data</code> includes a complete demo dataset for LEAO training (which is too large so we placed a download link), validation and testing. The test data also serve as the input to volume reconstruction.

- `./vol_reconstruction` includes the MATLAB codes for volume reconstruction of aberrated light-field data.

<hr>

<h2 id="Environment">2. Environment</h2>

Our code is built with PyTorch 2.2.1+cu121 and compatible on both Windows and Linux operating system.

To use our code, you should create a virtual environment and install the required packages first.

```
$ conda create -n LEAO python=3.12.2
$ conda activate LEAO
$ pip install -r requirements.txt
```

Also, follow the tutorials in [PyTorch Official Page](https://pytorch.org/get-started/locally/) to install the PyTorch that suits you.

After that, remember to install the right version of CUDA and cuDNN, if you want to use GPU. You can get the compatible version (e.g., cudatoolkit==11.3.1, cudnn==8.2.1) by searching

```
$ conda search cudatoolkit --info
$ conda search cudnn --info
```

then installing the corresponding version

```
$ conda install cudatoolkit==11.3.1
$ conda install cudnn==8.2.1
```

<hr>

<h2 id="Aberration estimation with LEAO">3. Aberration estimation with LEAO</h2>

With a trained model, you can perform inference by:

+ Prepare a folder of raw aberrated data. We provide a demo test dataset in `./demo_data/test_data`. 

+ Prepare a trained model with its weights saved in `.pth` format. We provide a demo model `./aberration_estimation/demo_model/epoch-best.pth`.

+ Prepare a configuration file that contains the raw data information and model information. We provide a demo `./aberration_estimation/demo_model/test.yaml`, which in default uses our demo data and model. A detailed explanation of all entries in the configuration file is listed below. 

+ Run `./aberration_estimation/test.py` after replacing the parameter `config` with your configuration file path, and `epoch_detail` with your trained model file name. Alternatively, specify the parameters and perform the execution in your terminal by running `python ./aberration_estimation/test.py --config [your_config_path] --epoch_detail [your_trained_model_file_name]`.

+ Estimated aberrations will be stored in a folder named `TestResult` inside your model folder. If you run `./aberration_estimation/test.py` defaultly without any modifications, the results will appear in `./aberration_estimation/demo_model/TestResult`.

If you would like to train a model of your own, follow these steps:

+ Prepare a folder of training data and validation data. We provide downloading link to demo training and validation dataset in `./demo_data/`, because file size is too large to upload directly to Github.

+ Prepare a configuration file. We provide a demo `./aberration_estimation/train.yaml`. A detailed explanation of all entries in the configuration file is listed below.

+ Run `./aberration_estimation/train.py` after replacing the parameter `config` with your configuration file path, and `gpu` with GPU IDs you want to use. Alternatively, specify the parameters and perform the execution in your terminal by running `python ./aberration_estimation/train.py --config [your_config_path] --gpu [gpu_id]`.

+ A folder will be created for each model you train, under the folder `./aberration_estimation/saved_models`. Weights, log information and a copy of the used configuration file will be saved in this folder.

+ Monitor the training by running `tensorboard --logdir [your_model_folder] --samples_per_plugin images=1000` and tracking the training losses, validation metrics and output aberrations via tensorboard.

Below is a table of all the parameters in a configuration file:

| **Section**       | **Key**               | **Description**                                                                                                                                                                                                       | **Example Value**                      |
| ----------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| **load_pretrain** |                       | Path to the pre-trained model. If you do not wish to load one, delete this line.                                                                                                                                      | `pretrained_model_1lambda.pth`         |
| **save_details**  | `subfolder`           | Subdirectory where a series of models can be saved.                                                                                                                                                                   | `mymodel`                              |
|                   | `type`                | Type or category of the data.                                                                                                                                                                                         | `Simulated_neurons`                    |
|                   | `suffix`              | Additional identifier for saved files.                                                                                                                                                                                | `train_num2000`                        |
| **train_dataset** | `load_num`            | Number of data to be loaded into memory for faster computation. Should not be larger than the number of training stacks.                                                                                              | `0` (do not load)                      |
|                   | `inp_path`            | List of paths to input training data.                                                                                                                                                                                 | `./training_data/input`                |
|                   | `gt_path`             | List of paths to ground truth (GT) training data.                                                                                                                                                                     | `../demo_data/training_data/gt/`       |
|                   | `zern_index`          | Order of Zernike polynomials.                                                                                                                                                                                         | `SH` or `ANSI`                         |
|                   | `minmax_norm`         | Flag to enable Min-Max normalization.                                                                                                                                                                                 | `0` (disabled) or `1` (enabled)        |
|                   | `adjustIntParam`      | Intensity adjustment parameter for input normalization.                                                                                                                                                               | `65535`                                |
|                   | `batch_size`          | Number of samples in each training batch.                                                                                                                                                                             | `1`                                    |
|                   | `sampler`             | Sampling strategy for training data.                                                                                                                                                                                  | `torch.utils.data.RandomSampler`       |
|                   | `cut`                 | Number of pixels to be cutted around each edge of the input data to discard unwanted regions.                                                                                                                         | `0`                                    |
|                   | `patch_size`          | Size of input patch to the network. Should not be larger than the raw data size. If given, crop the raw data to given size.                                                                                           | `280`                                  |
|                   | `mask_flag`           | Whether to use foreground masks for data cropping. Only valid when `patch_size` < raw data size. Set to 1 if you want to make sure areas with signals are cropped, at the cost of more computation when loading data. | `0` (no masks) or `1` (use masks)      |
|                   | `angle_num`           | Number of angular measurements used for training. Should not be larger than the number of angular measurements of raw data.                                                                                           | `113`                                  |
|                   | `desired_r`           | Desired radius within which angular measurements are valid.                                                                                                                                                           | `6`                                    |
|                   | `Nnum`                | Number of pixels in the x or y direction of the microlens array in light-field imaging.                                                                                                                               | `13`                                   |
| **val_dataset**   | Same as train_dataset | Validation dataset uses the same keys with paths pointing to validation data directories.                                                                                                                             | `../demo_data/val_data/input/`         |
| **model**         | `name`                | Name of the model architecture being used.                                                                                                                                                                            | `rcan_regressor2d_v7`                  |
|                   | `args.inp_angle`      | Number of input angles for the model. Should be the same as `angle_num`.                                                                                                                                              | `113`                                  |
|                   | `args.channel`        | Number of feature channels in the network.                                                                                                                                                                            | `64`                                   |
|                   | `args.reg_channel`    | Number of channels for estimator layers.                                                                                                                                                                              | `64`                                   |
|                   | `args.n_ResGroup`     | Number of residual groups in the model.                                                                                                                                                                               | `3`                                    |
|                   | `args.n_RCAB`         | Number of residual channel attention blocks per group.                                                                                                                                                                | `2`                                    |
| **optimizer**     | `name`                | Optimizer used for training.                                                                                                                                                                                          | `adam`                                 |
|                   | `args.lr`             | Learning rate for the optimizer.                                                                                                                                                                                      | `0.0001`                               |
| **multi_step_lr** | `milestones`          | Epochs at which the learning rate is reduced.                                                                                                                                                                         | `[50, 100, 150]`                       |
|                   | `gamma`               | Multiplicative factor for learning rate reduction.                                                                                                                                                                    | `0.7`                                  |
| **epoch_max**     |                       | Total number of training epochs.                                                                                                                                                                                      | `200`                                  |
| **epoch_val**     |                       | Frequency (in epochs) of validation runs.                                                                                                                                                                             | `1`                                    |
| **epoch_save**    |                       | Frequency (in epochs) of saving model checkpoints.                                                                                                                                                                    | `20`                                   |
| **recon_loss**    |                       | Reconstruction loss function used during training.                                                                                                                                                                    | `nn.MSELoss()`                         |
| **phase_loss**    |                       | Custom phase loss function used for optimization.                                                                                                                                                                     | `utils.zernike2phasemap_loss_SH`       |
| **latent_loss**   |                       | Latent space loss function for model regularization.                                                                                                                                                                  | `torch.nn.TripletMarginLoss(margin=1)` |

<hr>

<h2  id="Volume reconstruction">4. Volume reconstruction</h2>

After aberration estimation with LEAO, you can reconstruct unseen light-field data by running `./vol_reconstruction/main_recon.m`. Below are the important parameters in this script:

| **Parameter**     | **Description**                                                                                                                      | **Example Value**                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| `data_name`       | Name of the aberrated raw data file.                                                                                                 | `'test'`                                            |
| `maxIter`         | Reconstruction iteration number.                                                                                                     | `10`                                                |
| `Nnum`            | Number of pixels in the x or y direction of the microlens array in light-field imaging.                                              | `13`                                                |
| `save_iter`       | Frequency of saving (in iterations).                                                                                                 | `5`                                                 |
| `LEAO_model_path` | Path to the estimated aberrations.                                                                                                   | `'../aberration_estimation/demo_model/TestResult/'` |
| `epoch_detail`    | Name of the saved weights to be used, in case there are multiple estimations from different saved weights, usually different epochs. | `epoch-best`                                        |