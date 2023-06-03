model_root=params
data_name=$1
encoder=$2
batch_size=$3
base_lr=$4
num_tokens=$5
gate_init=$6
output_dir=results
model_type="ssl-vit"


if [ $data_name = "cub" ]
then
  data_path=./data/CUB_200_2011
  data_name="CUB"
  num_class=200
elif [ $data_name = "dogs" ] 
then
  data_path=./data/Dogs
  data_name="StanfordDogs"
  num_class=120
elif [ $data_name = "cars" ] 
then
  data_path=./data/Cars
  data_name="StanfordCars"
  num_class=196
elif [ $data_name = "nabirds" ] 
then
  data_path=./data/nabirds
  data_name="nabirds"
  num_class=555
elif [ $data_name = "flowers" ] 
then
  data_path=./data/Flowers
  data_name="OxfordFlowers"
  num_class=102
elif [ $data_name = "vtab-flowers" ] 
then
  data_path=~/datasets
  data_name="vtab-oxford_flowers102"
  num_class=102
elif [ $data_name = "sun397" ] 
then
  data_path=~/datasets
  data_name="vtab-sun397"
  num_class=397
elif [ $data_name = "pets" ] 
then
  data_path=~/datasets/
  data_name="vtab-oxford_iiit_pet"
  num_class=37
elif [ $data_name = "dmlab" ] 
then
  data_path=~/datasets
  data_name="vtab-dmlab"
  num_class=6
elif [ $data_name = "clevr-distance" ] 
then
  data_path=~/datasets
  data_name='vtab-clevr(task="closest_object_distance")'
  num_class=6
elif [ $data_name = "clevr-count" ] 
then
  data_path=~/datasets
  data_name='vtab-clevr(task="count_all")'
  num_class=8
elif [ $data_name = "caltech101" ] 
then
  data_path=./data
  data_name="vtab-caltech101"
  num_class=102
elif [ $data_name = "cifar100" ] 
then
  data_path=./data
  data_name="vtab-cifar(num_classes=100)"
  num_class=100
elif [ $data_name = "dsprites-orientation" ] 
then
  data_path=~/datasets
  data_name='vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)'
  num_class=16
elif [ $data_name = "dsprites-location" ] 
then
  data_path=~/datasets
  data_name='vtab-dsprites(predicted_attribute="label_x_position",num_classes=16)'
  num_class=16
elif [ $data_name = "dtd" ] 
then
  data_path=./data
  data_name="vtab-dtd"
  num_class=47
elif [ $data_name = "eurosat" ] 
then
  data_path=~/datasets
  data_name="vtab-eurosat"
  num_class=10
elif [ $data_name = "resisc" ] 
then
  data_path=~/datasets
  data_name="vtab-resisc45"
  num_class=45
elif [ $data_name = "smallnorb-azimuth" ] 
then
  data_path=~/datasets
  data_name='vtab-smallnorb(predicted_attribute="label_azimuth")'
  num_class=18
elif [ $data_name = "smallnorb-elevation" ] 
then
  data_path=~/datasets
  data_name='vtab-smallnorb(predicted_attribute="label_elevation")'
  num_class=9
elif [ $data_name = "patch" ] 
then
  data_path=~/datasets
  data_name="vtab-patch_camelyon"
  num_class=2
elif [ $data_name = "kitti" ] 
then
  data_path=~/datasets
  data_name='vtab-kitti(task="closest_vehicle_distance")'
  num_class=4
elif [ $data_name = "svhn" ] 
then
  data_path=~/datasets
  data_name="vtab-svhn"
  num_class=10
elif [ $data_name = "retino" ]
then
  data_path=~/datasets
  data_name='vtab-diabetic_retinopathy(config="btgraham-300")'
  num_class=5
fi


seed=42
echo $data_name
echo $data_path
echo $encoder


python3 train.py \
        --config-file configs/base-prompt.yaml \
        DATA.BATCH_SIZE "${batch_size}" \
        DATA.CROPSIZE "224" \
        MODEL.PROMPT.NUM_TOKENS "${num_tokens}" \
        SOLVER.WEIGHT_DECAY "0.0" \
        SOLVER.BASE_LR "${base_lr}" \
        MODEL.PROMPT.DROPOUT "0.1" \
        SEED ${seed} \
        MODEL.TYPE "${model_type}" \
        MODEL.PROMPT.DEEP "False" \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}" \
        DATA.NAME "${data_name}" \
        DATA.FEATURE "${encoder}" \
        DATA.NUMBER_CLASSES "${num_class}" \
        MODEL.TRANSFER_TYPE "prompt" \
        MODEL.PROMPT.INITIATION "random" \
        MODEL.PROMPT.TEMP_LEARN "True" \
        MODEL.PROMPT.GATE_PRIOR "True" \
        MODEL.PROMPT.GATE_INIT "${gate_init}" \
        MODEL.PROMPT.VIT_POOL_TYPE "original" \
        OUTPUT_DIR "${output_dir}/seed${seed}" \
        
