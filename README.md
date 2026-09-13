# GAN and WGAN image generation behind a Flask app

> **中文简介**
>
> 一个用 Flask 做的二次元头像生成网页，GAN、DCGAN、WGAN-CP、WGAN-GP 四种模型共用同一个界面。训练数据是 14 万张 128×128 的二次元头像，共 3.7 GB。三个卷积模型用 Inception Score 做了横向评测，结论在下面。训练好的 `.pkl` 权重不在仓库里，要自己先训练；页面模板来自 HTML5 UP 的 Massively。

## What this was built for

The task was anime avatar generation: train a generator that produces 128×128 portraits good enough to be worth looking at, and then compare the three convolutional variants people argue about. The dataset is roughly 140,000 avatars at 128×128 or 96×96, about 3.7 GB on disk. No labels, since GAN training is unsupervised; the images are only cut to size and normalised before going into the loader.

Four models sit behind one dropdown rather than four separate scripts, so the same UI can drive any of them and the comparison stays honest.

## Four models behind one dropdown

Keeping four implementations around only makes sense if they are actually different, so it is worth being precise about how they differ here.

`models/gan.py` is the fully connected one. The generator is `Linear(100, 256) -> LeakyReLU(0.2) -> Linear(256, 512) -> LeakyReLU -> Linear(512, 1024) -> Tanh`, and the discriminator mirrors that back down to a single `Sigmoid` unit. It never convolves, so it does not care about image shape at all: the discriminator flattens its input with `images.view(self.batch_size, -1)` before the first linear layer, which is why the code path only works with a fixed batch size. BCE loss, Adam at `lr=0.0002` with `weight_decay=0.00001`, latent sampled as a plain `(batch, 100)` tensor.

`models/dcgan.py` is the convolutional version and the one most people actually train. The generator goes `ConvTranspose2d(100 -> 1024, kernel_size=4, stride=1, padding=0)` and then 1024 -> 512 -> 256 -> channels with `kernel_size=4, stride=2, padding=1`, `BatchNorm2d` between the blocks and a `Tanh` at the end. The discriminator is the mirror image with convolutions. BCE loss again, Adam at `lr=0.0002` with `betas=(0.5, 0.999)`, and the latent is shaped `(batch, 100, 1, 1)`.

`models/wgan_clipping.py` reuses DCGAN's generator and discriminator shapes but throws away the sigmoid. The critic outputs an unbounded score, the loss is `d_loss_fake - d_loss_real`, and the generator maximizes `D(G(z))` instead of fooling a classifier. After every critic step the weights are clamped to `[-0.01, 0.01]` with `p.data.clamp_()`, and the optimizer is RMSprop at `lr=0.00005`, which is what the original paper prescribes and the comment in the file says out loud. Five critic updates per generator update.

`models/wgan_gradient_penalty.py` swaps clipping for the penalty, which is the fix for everything clipping does badly. Adam this time, `lr=1e-4` with `betas=(0.5, 0.999)`, and `lambda_term = 10` multiplying `((gradients.norm(2, dim=1) - 1) ** 2).mean()` on interpolated samples. The critic uses `InstanceNorm2d(..., affine=True)` where the other two use `BatchNorm2d`, since batch norm mixes statistics across the batch and breaks a per-sample gradient penalty. Also five critic updates per generator step.

|  | GAN | DCGAN | WGAN-CP | WGAN-GP |
| --- | --- | --- | --- | --- |
| Optimizer | Adam `2e-4`, `weight_decay=1e-5` | Adam `2e-4`, betas `(0.5, 0.999)` | RMSprop `5e-5` | Adam `1e-4`, betas `(0.5, 0.999)` |
| Objective | BCE | BCE | Wasserstein, weight clipping `0.01` | Wasserstein, gradient penalty `10` |
| Critic steps per generator step | 1 | 1 | 5 | 5 |
| Latent shape | `(n, 100)` | `(n, 100, 1, 1)` | `(n, 100, 1, 1)` | `(n, 100, 1, 1)` |
| Normalisation in critic | none | `BatchNorm2d` | `BatchNorm2d` | `InstanceNorm2d(affine=True)` |
| Batch size | `--batch_size` | `--batch_size` | `64`, hardcoded | `64`, hardcoded |

## Measuring them

Inception Score is the metric, computed in `utils/inception_score.py`: push generated images through a pretrained `inception_v3`, take the 1000-way softmax, and score `exp(E[KL(p(y|x) || p(y))])` over ten splits. The training loop in each class calls it every `SAVE_PER_TIMES` iterations on 800 freshly sampled latents with `batch_size=16`, and separately saves a 64-image grid to `training_result_images/img_generator_iter_<n>.png` so the run can be watched as it goes.

Over 40,000 generator iterations the three separate cleanly and in an order that is not the one you would guess from the theory:

DCGAN finishes highest at about 5.7, WGAN-GP sits below it at about 5.5, and WGAN-CP trails at about 4.9 with visibly white-speckled samples.

The gap between the two WGANs is the expected result and has a nameable cause. Clipping the critic weights to a hard box is a crude way to bound the Lipschitz constant, and it pushes the critic toward saturated, low-information outputs; the gradient penalty bounds it smoothly instead, and the artefacts go with it. That is the whole argument for WGAN-GP over WGAN-CP, and it shows up here as roughly half a point of IS plus a visible difference in the images.

The DCGAN result is the interesting one, because the highest score is not the best model. DCGAN has no Lipschitz constraint at all, so its generated distribution is narrower and its outputs are easier for Inception-V3 to sort confidently into a single class. That inflates `p(y|x)` concentration and therefore IS, without the images being better, and DCGAN is visibly the least stable of the three during training. A high IS with a low variety of outputs is a known failure mode of the metric, and this run is a clean example of it. WGAN-GP is the model to keep.

Nothing here was free: the first attempt ran batch size 64 at 32×32 and the images were too blurry to evaluate; dropping to batch 16 at 64×64 made each step too noisy for the networks to pick up features; the run that produced the curve above is batch 64 at 64×64, with the generator and discriminator depth adjusted to match the larger resolution.

## What all four share

The constructor signature is identical across the four classes: `is_train`, download, dataroot, dataset, generator_iters, cuda, batch_size, load_D, load_G`. So is the method surface: `trainloader`, `evaluatertest_loader, D_path, G_path`, `save_model`, `load_model`, `generate_img`, `generate_latent_walk`. `save_model` writes `./generator.pkl` and `./discriminator.pkl` into the process working directory, and every `train()` calls it at the end of the run. The two WGAN classes also checkpoint on a `SAVE_PER_TIMES` interval, 1000 iterations for the clipping model and 100 for the gradient-penalty one, while `GAN.train()` saves every 1000 generator iterations and `DCGAN.train()` saves once per epoch.

`utils/feature_extraction_test.py` does a second, cheaper evaluation: it pulls discriminator features and fits a `LogisticRegression` on top. `utils/tensorboard_logger.py` logs scalars, images and histograms into `./logs`, which is the only reason a PyTorch project has a TensorFlow dependency.

## Training one

There is no `train.py`, and that is the main practical obstacle. You instantiate the class yourself, get a loader from `utils.data_loader.get_data_loader(dataset, batch_size, dataroot, download)` and call `train()` on it. `utils/config.py` defines the argparse you would wire up to that: `--model` (choices `GAN`, `DCGAN`, `WGAN-CP`, `WGAN-GP`), `--dataroot`, `--dataset` (`mnist`, `fashion-mnist`, `cifar`, `stl10`, `animedata`), `--epochs`, `--batch_size`, `--cuda`, `--load_D`, `--load_G`, `--generator_iters`, plus `--is_train` and `--download` as strings. `check_args()` also sets `args.channels` to 3 for cifar, stl10 and animedata and to 1 otherwise, and turns the `--cuda` string into a bool.

`animedata` means a plain image folder, handled by `utils/get_date.py`, which walks `root/train` and `root/test` and picks up `.png` and `.jpg` files. That is the path this project used. `utils/fashion_mnist.py` carries local MNIST and FashionMNIST dataset classes adapted from the torchvision source, so `mnist` and `fashion-mnist` do not go through `torchvision.datasets` at all.

## The Flask side

`python main.py` starts it on port 5000 with `debug=True`. `/` and `/index` return the landing page, `/generic` and `/elements` return template pages, and `/upload` drops a posted file into `./images`.

`/test` is the route that does work. It reads a JSON body containing `model`, `is_train`, `download`, `dataroot`, `dataset`, `generator_iters`, `cuda`, `batch_size`, `load_D` and `load_G`, builds the matching class, runs `evaluater(test_loader, load_D, load_G)` and re-renders `generic.html`. Before construction, `load_D` and `load_G` are rewritten to `<stem><model>.pkl`, so one checkpoint pair per model can sit next to each other without collisions.

The image the page shows comes out of `evaluater()`: DCGAN, WGAN-CP and WGAN-GP all sample a 64-image grid and `utils.save_image` it straight to `static/images/dgan_model_image.png`, which is the file `/generic` and `/test` pass to the template. `GAN.evaluater()` writes its grid to `gan_model_image.png` in the working directory instead, so selecting plain GAN leaves the page showing whatever `dgan_model_image.png` happens to be on disk from a previous run.

`/login` and `/register` connect through `mysql.connector` and query a `user (name, password)` table. Both hardcode `database='c4'` while taking host, user and password from `MYSQL_HOST`, `MYSQL_USER` and `MYSQL_PASSWORD`. `config.py` builds the SQLAlchemy URI from a wider set (`MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`, default `C4`) using the `pymysql` driver, and `main.py` creates a `SQLAlchemy(app)` from it even though the auth routes do not use it. Point both paths at the same schema.

## The front end

`templates/` and `static/assets/` are the HTML5 UP "Massively" template by [@ajlkn](https://html5up.net), free for personal and commercial use under CCA 3.0. The demo copy has been swapped for the four model names, so the landing page tiles read GAN, DCGAN, WGAN-CP and WGAN-GP.

MIT, see `LICENSE`. Copyright (c) 2026 NDCGT.
