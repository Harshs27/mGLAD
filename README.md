## mGLAD  
A meta learning based approach to recover sparse graphs. This work proposes `mGLAD` which is a unsupervised version of a previous `GLAD` model (GLAD: Learning Sparse Graph Recovery (ICLR 2020 - [link](<https://openreview.net/forum?id=BkxpMTEtPB>)).  

Key Benefits & features:  
- It is a fast alternative to solving the Graphical Lasso problem as
    - GPU based acceleration can be leveraged
    - Requires less number of iterations to converge due to neural network based acceleration of the unrolled optimization algorithm (Alternating Minimization).     
- mGLAD automatically learns the sparsity related regularization parameters. This gives an added benefit to mGLAD over other graphical lasso solvers.  
- The meta loss is the logdet objective of the graphical lasso `1/B(-1*log|theta|+ <S, theta>)`, where `B=batch_size, S=input covariance matrix, theta=predicted precision matrix`.   

## Setup  
The `setup.sh` file contains the complete procedure of creating a conda environment to run mGLAD model. In case of dependencies conflict, one can alternatively use this command `conda env create --name mGLAD --file=environment.yml`.  

## demo-mGLAD notebook  
A minimalist working example of mGLAD. It is a good entry point to understand the code structure as well as the GLAD model.  

## Citation
If you find this method useful, kindly cite the following 2 associated papers:

- mGLAD:  

- GLAD:  
@article{shrivastava2019glad,
  title={GLAD: Learning sparse graph recovery},
  author={Shrivastava, Harsh and Chen, Xinshi and Chen, Binghong and Lan, Guanghui and Aluru, Srinvas and Liu, Han and Song, Le},
  journal={arXiv preprint arXiv:1906.00271},
  year={2019}
}
