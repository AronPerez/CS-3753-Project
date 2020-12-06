# CS-3753-Project
- Aaron Perez
- Mykel Roa
- David Levy

# Overview
The framework consists of three modules:
1. **Datasets** - data available for training ML models. [Access to dataset](https://mega.nz/folder/oK4EVaBD#3Epl63VEuzxfI2wNoijG3g)
2. **L5Kit** - the core library supporting functionality for reading the data and framing planning and simulation problems as ML problems.
3. **Examples** - an ever-expanding collection of jupyter notebooks which demonstrate the use of L5Kit to solve various AV problems.

# Couple of things to note

- **model_architecture**: you can put 'resnet18', 'resnet34' or 'resnet50'. For the pretrained model we use resnet18 so we need to use 'resnet18' in the config.
- **weight_path**: path to the pretrained model. If you don't have a pretrained model and want to train from scratch, put weight_path = False.
- **model_name**: the name of the model that will be saved as output, this is only when train= True.
- **train**: True if you want to train the model.
- **predict**: True if you want to predict and submit to Kaggle.
- **lr**: learning rate of the model.
- **raster_size**: specify the size of the image, the default is [224,224]. Increase raster_size can improve the score. However the training time will be significantly longer.
- **batch_size**: number of samples for one forward pass
- **steps**: number of batches of data that the model will be trained on. (note this is not epoch)
- **checkpoint_every_n_steps**: the model will be saved at every n steps, again change this number as to how you want to keep track of the model.
Note (Louis): The original pretrained model doesn't save the state of optimizer, so continute training doesn't work too well.

# Scoring
For scoring, we calculate the negative log-likelihood of the ground truth data given these multi-modal predictions. Let us take a closer look at this. Assume, ground truth positions of a sample trajectory are

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20x_1%2C%20%5Cldots%2C%20x_T%2C%20y_1%2C%20%5Cldots%2C%20y_T),

and we predict K hypotheses, represented by means

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%5Cbar%7Bx%7D_1%5Ek%2C%20%5Cldots%2C%20%5Cbar%7Bx%7D_T%5Ek%2C%20%5Cbar%7By%7D_1%5Ek%2C%20%5Cldots%2C%20%5Cbar%7By%7D_T%5Ek).

In addition, we predict confidences c of these K hypotheses.
We assume the ground truth positions to be modelled by a mixture of multi-dimensional independent Normal distributions over time,
yielding the likelihood

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20p%28x_%7B1%2C%20%5Cldots%2C%20T%7D%2C%20y_%7B1%2C%20%5Cldots%2C%20T%7D%7Cc%5E%7B1%2C%20%5Cldots%2C%20K%7D%2C%20%5Cbar%7Bx%7D_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7B1%2C%20%5Cldots%2C%20K%7D%2C%20%5Cbar%7By%7D_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7B1%2C%20%5Cldots%2C%20K%7D%29)

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%3D%20%5Csum_k%20c%5Ek%20%5Cmathcal%7BN%7D%28x_%7B1%2C%20%5Cldots%2C%20T%7D%7C%5Cbar%7Bx%7D_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7Bk%7D%2C%20%5CSigma%3D1%29%20%5Cmathcal%7BN%7D%28y_%7B1%2C%20%5Cldots%2C%20T%7D%7C%5Cbar%7By%7D_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7Bk%7D%2C%20%5CSigma%3D1%29)

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%3D%20%5Csum_k%20c%5Ek%20%5Cprod_t%20%5Cmathcal%7BN%7D%28x_t%7C%5Cbar%7Bx%7D_t%5Ek%2C%20%5Csigma%3D1%29%20%5Cmathcal%7BN%7D%28y_t%7C%5Cbar%7By%7D_t%5Ek%2C%20%5Csigma%3D1%29)

yielding the loss

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20L%20%3D%20-%20%5Clog%20p%28x_%7B1%2C%20%5Cldots%2C%20T%7D%2C%20y_%7B1%2C%20%5Cldots%2C%20T%7D%7Cc%5E%7B1%2C%20%5Cldots%2C%20K%7D%2C%20%5Cbar%7Bx%7D_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7B1%2C%20%5Cldots%2C%20K%7D%2C%20%5Cbar%7By%7D_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7B1%2C%20%5Cldots%2C%20K%7D%29)

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%3D%20-%20%5Clog%20%5Csum_k%20e%5E%7B%5Clog%28c%5Ek%29%20&plus;%20%5Csum_t%20%5Clog%20%5Cmathcal%7BN%7D%28x_t%7C%5Cbar%7Bx%7D_t%5Ek%2C%20%5Csigma%3D1%29%20%5Cmathcal%7BN%7D%28y_t%7C%5Cbar%7By%7D_t%5Ek%2C%20%5Csigma%3D1%29%7D)

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%3D%20-%20%5Clog%20%5Csum_k%20e%5E%7B%5Clog%28c%5Ek%29%20-%5Cfrac%7B1%7D%7B2%7D%20%5Csum_t%20%28%5Cbar%7Bx%7D_t%5Ek%20-%20x_t%29%5E2%20&plus;%20%28%5Cbar%7By%7D_t%5Ek%20-%20y_t%29%5E2%7D)

You can find our implementation [here](https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py#L4), which uses *error* as placeholder for the exponent

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20L%20%3D%20-%5Clog%20%5Csum_k%20e%5E%7B%5Ctexttt%7Berror%7D%7D)

and for numeral stability further applies the [log-sum-exp trick](https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations):
Assume, we need to calculate the logarithm of a sum of exponentials:

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20LSE%28x_1%2C%20%5Cldots%2C%20x_n%29%20%3D%20%5Clog%28e%5E%7Bx_1%7D%20&plus;%20%5Cldots%20&plus;%20e%5E%7Bx_n%7D%29)

Then, we rewrite this by substracting the maximum value x<sup>*</sup> from each exponent, resulting in much increased numerical stability:

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20LSE%28x_1%2C%20%5Cldots%2C%20x_n%29%20%3D%20x%5E*%20&plus;%20%5Clog%28e%5E%7Bx_1%20-%20x%5E%7B*%7D%7D%20&plus;%20%5Cldots%20&plus;%20e%5E%7Bx_n%20-%20x%5E%7B*%7D%7D%29)
