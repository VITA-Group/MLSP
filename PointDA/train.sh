CUDA_VISIBLE_DEVICES=0
DefRec_on_trgt=False
DefRec_weight=0.5
Normal_pred_weight=0.5
Density_weight=0.05

Norm_on_trgt=False
Density_on_trgt=False
Density_normal_viainput_onsrc=False
Density_normal_viainput=True
Density_normal_defpart=False
Normal_ondef=True
Density_ondef=True

sourcedata=scannet
targetdataset=modelnet
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python trainer.py  --out_path ../../DAexperiments/dgcnn/sup/$sourcedata'_'$targetdataset/  --Density_normal_viainput_onsrc=$Density_normal_viainput_onsrc --dataroot ../../ --src_dataset=$sourcedata --trgt_dataset=$targetdataset  --DefRec_on_trgt=$DefRec_on_trgt --DefRec_weight=$DefRec_weight --Density_normal_viainput=$Density_normal_viainput --Normal_ondef=$Normal_ondef --Density_ondef=$Density_ondef --Density_weight=$Density_weight --Density_normal_defpart=$Density_normal_defpart --Density_on_trgt=$Density_on_trgt --Norm_on_trgt=$Norm_on_trgt --normal_pred_weight=$Normal_pred_weight
###modelfile is the file that you save from last step
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_spst.py  --out_path ../../  --dataroot ../../ --src_dataset=$sourcedata --trgt_dataset=$targetdataset  --round=2  --epochs=20 --model_file=$modelfile --threshold=1.5492