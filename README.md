# NMSSM_CosmoTrans

Computes the phase transitions in the Next-to-Minimal Supersymmetric Standard Model (NMSSM) using [CosmoTransitions](https://github.com/clwainwright/CosmoTransitions) (see also [this link](https://clwainwright.net/CosmoTransitions/)). 

This code was used to produce the results for the publication
[Nucleation is More than Critical -- A Case Study of the Electroweak Phase Transition in the NMSSM](https://arxiv.org/abs/2009.10743)
(Sebastian Baum, Marcela Carena, Nausheen R. Shah, Carlos E. M. Wagner, and Yikun Wang).

This code needs the python package [CosmoTransitions](https://github.com/clwainwright/CosmoTransitions). It was tested with [CosmoTransitions_2.0.5](https://pypi.org/project/cosmoTransitions/2.0.5/#files) and **python2.7**.

If you use **NMSSM_CosmoTrans** for your work, please cite [arXiv:2009.10743](https://arxiv.org/abs/2009.10743).

# Description

The main code is **NMSSM_CosmoTrans/NMSSM_potential_CT.py**, implementing the effective (one-loop + Daisy resummation) potential of the Next-to-Minimal Supersymmetric Standard Model in CosmoTransitions. 
Please see [arXiv:2009.10743](https://arxiv.org/abs/2009.10743) for the details of the implementation. 
The naming conventions for the functions in **NMSSM_CosmoTrans/NMSSM_potential_CT.py** are following the default conventions in [CosmoTransitions](https://clwainwright.net/CosmoTransitions/) as closely as possible.

Examples for the usage of NMSSM_CosmoTrans/NMSSM_potential_CT.py can be found in NMSSM_CosmoTrans/Examples.
- **NMSSM_CosmoTrans/Examples/RunScan.py** is an example script showing how to run a parameter scan
- **NMSSM_CosmoTrans/Examples/ClassifyPoints.py** is an example script showing how to classify the phase transition patterns of the results of **RunScan.py**
- the folder **NMSSM_CosmoTrans/data/Scan_EXAMPLE** contains result files from a run of **RunScan.py**




