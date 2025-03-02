# ThinCrackIdentification
TCI-NET
# Abstract:
Accurate identification of thin cracks on concrete surfaces is a key technology for the automated maintenance of structures, with widespread applications in intelligent inspection tasks such as roads, bridges, and building facades. Existing crack identification methods, based on encoder-decoder architectures, suffer from inaccuracies and insufficient reliability for diverse concrete surface cracks. This study proposes a reliable crack identification method using the U-Net semantic segmentation network, incorporating image structural feature enhancement and multi-level consistency constraints. The method builds on a pre-trained multi-scale semantic feature encoding network, integrating various crack edge structure information extraction operators, and constructs a lightweight crack spatial detail feature extraction module. Through an attention mechanism, the extracted multi-scale semantic features and high-resolution structural features are adaptively fused, enabling reliable and intelligent identification of thin cracks in complex environments. Extensive experiments on the large-scale CrackSeg9K crack recognition public dataset demonstrate that, compared to the U-Net method, the proposed approach achieves significant improvements in accuracy, with more refined crack extraction results and higher cross-scene generalization of the model.

# The Datasource
①The dataset with 160 typical pictures of cracks on concrete surfaces.
    Baidu Netdisk Link: https://pan.baidu.com/s/11eQrNQ_fzrrBMBiUxXHuhw. Password: s5h5 
    
②Crackseg9k: A Collection of Crack Segmentation Datasets.
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGIEBY.
