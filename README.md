### Evaluate INSTR on STIOS
STIOS is a table-top object dataset recorded by two stereo sensors with manual annotated instances for each frame ([website](https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-17628/#gallery/36367), [code utilities](https://github.com/DLR-RM/stios-utils)).
To evaluate and reproduce the experiments in the paper (Tab. 2), [download STIOS](https://zenodo.org/record/4706907#.YROCeDqxVhE) and the [pretrained model](https://drive.google.com/uc?id=1wFSTa5IoJYUTYpGeunE7SzXMfqRWXHi7).
Extract the pretrained model in the project's root directory.
Then, run `python predict_stios.py --state-dict pretrained_instr/models/pretrained_model.pth --root /path/to/stios {--rcvisard, --zed}`.
This will generate mIoU and F1 scores for every scene.

### Demo
Download the [pretrained model](https://drive.google.com/uc?id=1wFSTa5IoJYUTYpGeunE7SzXMfqRWXHi7) and extract the contents here: `./pretrained_instr/models/`.
Overwrite the [camera class](demo.py) so that it returns a pair of stereo images (RGB, np.array, uint8) from your stereo camera.
Then, run `python demo.py` for the default demo.

Run `python demo.py --help` or have a look at the [predictor class](predictor.py) for further information.

### DINO demo
Before trying the model will be needed to download the model instr and to create a virtual environment with conda using the `instr-dino.yml` with the following command:
`conda env create --name instr-dino --file=instr-dino.yml && conda activate instr-dino`
to be able to use the dino backbone and to see the result of the overlay mask run 
`python instr_dino_demo.py --root ./STIOS --rcvisard --save-dir ./new_test --backbone dinov2l --state-dict pretrained_instr/models/pretrained_model.pth`

in order to test it with other model change the `--backbone` flag with INSTR, dinov2b or dinov2l in order to use instr or ViT-B/14 distilled or ViT-L/14 distilled.

in order to evaluate use predict_stios like before but adding the flacg backbone with instr, dinov2l or dinov2b
`python predict_stios.py --state-dict pretrained_instr/models/model_39.pth --root STIOS --rcvisard --backbone dinov2b`