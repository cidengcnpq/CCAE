# <p align="center">Conditional Convolutional Autoencoder with Multi-Task Learning for Context-Driven Unsupervised Structural Assessment


#### Victor Higino Meneguitte Alves[^1], M.Sc. Student, Civil Engineer
-	Graduate Program in Civil Engineering, Faculty of Engineering, University of Juiz de Fora, Juiz de Fora, Minas Gerais, Brazil

#### Alexandre Abrahão Cury[^2], Associate Professor
-	Graduate Program in Civil Engineering, Faculty of Engineering, University of Juiz de Fora, Juiz de Fora, Minas Gerais, Brazil

[^1]: Orcid: [0000-0001-8959-050X](https://orcid.org/0000-0001-8959-050X)
[^2]: Orcid: [0000-0002-7199-6533](https://orcid.org/0000-0002-7199-6533)

## Abstract
<p align="justify">Modern Vibration-based Structural Health Monitoring (VSHM) increasingly relies on unsupervised Deep Learning (DL) strategies to detect damage without the need for previous known data. While Autoencoders (AE) are widely used for this purpose, standard architectures often process sensor data in isolation, failing to capture the spatial dependencies and contextual relationships inherent in multi-sensor networks. To address this gap, this paper proposes a Conditional Convolutional AutoEncoder (CCAE) within a Multi-Task Learning framework. The proposed methodology transforms raw 1D acceleration signals into 2D Continuous Wavelet Transform (CWT) scalograms, allowing convolutional layers to extract time-frequency features. The CCAE is trained with a dual objective: to accurately reconstruct the input scalogram and, simultaneously, to identify the signal’s spatial origin (sensor location/ unique ID). This auxiliary classification task is self-supervised and generates a dynamic contextual embedding that conditions the latent space, forcing the model to learn representations that are both structurally descriptive and spatially aware. For anomaly localization, a hierarchical framework is introduced, utilizing the Mahalanobis Distance (MD) for sensor-level quantification and a multivariate Kernel Density Estimation (KDE) for system-level data fusion. The methodology is validated on the Yellow Frame laboratory benchmark and the Z24 Bridge. Results demonstrate that conditioning the generative process on spatial context significantly enhances the distinctiveness of the feature space.

#### <p align="justify">Keywords: `Structural Health Monitoring`, `Self-supervised Learning`, `Convolutional AutoEncoder`, `Multi-Task Learning` and `Continuous Wavelet Transform`.

  ### **<p align="center">Graphical abstract: Overview of CCAE model training architecture**<img width="1280" height="945" alt="CCAE_architecture" src="https://github.com/user-attachments/assets/4a11d42a-27c7-4499-8c6a-6cfd2b8f6ab6" />

  ### **<p align="center">Graphical abstract: Flowchart of the proposed hierarchical anomaly detection framework**<img width="1280" height="602" alt="CCAE_architecture2" src="https://github.com/user-attachments/assets/898a4997-d5f2-48b4-acec-fc7c6783cf86" />

 ### Highlights
- [x] Real scale structure application
- [X] Unsupervised
- [X] Damage detection (Rytter scale I[^3])
- [X] Damage localization (Rytter scale II[^3])
      
###  **● Code for model optimization and training are on files:**
`Optimization_and_training_yellow_frame.py', 'Optimization_and_training_Z24.py'

###  **● Code for results visualization are on files:**
`Results_boxplot_yellow_frame.py', 'Results_boxplot_Z24.py'

###  **● Code for Yellow Frame signal data pre-processing in MATLAB is on file:**'Generate_images_CWT_yellow_frame.m'

###  **● Code for Z24 bridge signal data pre-processing in MATLAB are on files:**
`Generate_Raw_data_Z24_Alves_et_al_2024.m', 'generate_image_CWT_Z24.m'
[^3]: [A. Rytter, Vibrational based inspection of civil engineering structures. Dept. of Building Technology and Structural Engineering, Aalborg University, Fracture and Dynamics. R9314 (44) (1993).](https://vbn.aau.dk/en/publications/vibrational-based-inspection-of-civil-engineering-structures)
