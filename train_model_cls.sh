#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "No arguments provided. Please provide an argument (1-10)."
    exit 1
fi


# Get the argument
ARG=$1
model=$2
local_train=$3


echo "Model Number: $ARG"
echo "Backbone: $model"



# Define the base command
BASE_CMD="python -u Train.py"



DATA_DIR="--data_dir /mnt/mywork/data"



echo "Where training: $local_train"




# Run the command based on the argument
case $ARG in
    1)
        $BASE_CMD  --xmnet                --cls_included   --backbone_class  $model --dataset soybean_2_1
        ;;

    2)
        $BASE_CMD  --xmnet --maskguided   --cls_included      --backbone_class  $model --dataset soybean_2_1
        ;;

   3)
        $BASE_CMD                         --cls_included     --backbone_class  $model --dataset soybean_2_1
        ;;


    *)
        echo "Invalid argument. Please provide a number between 1 and 10."
        exit 1
        ;;
esac

echo "Command executed."
