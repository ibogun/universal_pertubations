
#!/bin/bash
CHECKPOINT_DIR=checkpoints
mkdir ${CHECKPOINT_DIR}
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
tar -xvf inception_resnet_v2_2016_08_30.tar.gz
mv inception_resnet_v2_2016_08_30.ckpt ${CHECKPOINT_DIR}
rm inception_resnet_v2_2016_08_30.tar.gz
