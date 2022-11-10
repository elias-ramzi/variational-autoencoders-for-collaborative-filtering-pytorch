python demo.py train --dataset_name ml-latest-small --processed_dir ml-latest-small/ --n_items 8157 --train_batch_size 100 --arch_type MultiVAE --runs 10 --gpu 0 > output/ml_mult_vae.out

python demo.py train --dataset_name ml-latest-small --processed_dir gowalla/ --n_items 40981 --train_batch_size 512 --arch_type MultiVAE --runs 10 --gpu 0 > output/gowalla_mult_vae.out

python demo.py train --dataset_name ml-latest-small --processed_dir yelp2018/ --n_items 38048 --train_batch_size 512 --arch_type MultiVAE --runs 10 --gpu 0 > output/yelp2018_mult_vae.out
