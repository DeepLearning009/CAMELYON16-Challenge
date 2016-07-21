How to train models?

VGG-16 training
1. python CorrTxtMaker.py
    Generate coordinates of VGG patch-candidates and them will be written into "./VGGSnapshots_L1/Tumor_XXX_coor.txt"

2. python VGGRandomTrainer_L1.py
    Train the VGG-16 Net, the trained models will be stored in  "./VGGSnapshots_L1/Camelyon_VGG_L1.caffemodel"

Res-152 training
1. Goto "Camelyon_VGGRes/TestTrainData/VGGDetecter/". 
   Copy the trained VGG-model from "./VGGSnapshots_L1/Camelyon_VGG_L1.caffemodel" into "VGGDetecter/Deploys/Camelyon_VGG_L1.caffemodel" 
   Generate the prediction results. The VGG results will be stored in "VGGDetecter/VGGResults/Tumor_XXX.csv" 

2. Copy the VGG results from "VGGDetecter/VGGResults/Tumor_XXX.csv" to "./ResSnapshots_L0/Tumor_XXX.csv".

3. python ResRandomTrainer_L0.py
     Train the ResNet-152, the trained models will be stored in "./ResSnapshots_L0/Camelyon_Res_L0.caffemodel"
