#!/bin/bash

dset=$1
dset=${dset:="wikipedia"}
# echo $dset

if [ ! -d $dset ]; then
    mkdir $dset
fi
cd $dset

case $dset in
wikipedia)
    SRC_P=/home/dataset/wikipedia_dataset
    FILE=(images.vgg19.mat texts.wiki.doc2vec.300.mat labels.wiki.mat \
          class_emb.wikipedia.Gnews-300d.mat)
    for f in ${FILE[@]}; do
        # echo $f
        ln -s $SRC_P/$f
    done

    mkdir disjoint
    cd disjoint
    SPLIT_P=/home/tom/codes/test.zsr/single-dataset-disjoint/data/wikipedia
    for i in `seq 0 5`; do
        ln -s $SPLIT_P/split-$i-DADN
    done
    ;;
pascal-sentences)
    SRC_P=/home/dataset/pascal-sentences
    FILE=(images.pascal-sentences.vgg19.4096d.mat \
          texts.pascal-sentences.doc2vec.300.mat \
          labels.pascal-sentences.mat \
          class_emb.pascal-sentences.Gnews-300d.mat)
    for f in ${FILE[@]}; do
        # echo $f
        ln -s $SRC_P/$f
    done
    ;;
*)
    echo Not implemented
    exit
    ;;
esac
