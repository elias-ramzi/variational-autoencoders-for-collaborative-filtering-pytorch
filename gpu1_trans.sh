# python demo.py train --dataset_name ml-latest-small --processed_dir ml-latest-small-trans/ --n_items 8157 --train_batch_size 100 --arch_type MultiDAE --runs 10 --gpu 1 > output/trans_ml_mult_dae.out

python demo.py train --dataset_name ml-latest-small --processed_dir gowalla_trans/ --n_items 40981 --train_batch_size 512 --arch_type MultiDAE --runs 10 --gpu 1 > output/trans_gowalla_mult_dae.out

python demo.py train --dataset_name ml-latest-small --processed_dir yelp2018_trans/ --n_items 38048 --train_batch_size 512 --arch_type MultiDAE --runs 10 --gpu 1 > output/trans_yelp2018_mult_dae.out
