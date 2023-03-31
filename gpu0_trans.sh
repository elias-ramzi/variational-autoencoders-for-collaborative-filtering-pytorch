python demo.py train --dataset_name ml-latest-small --processed_dir ml-latest-small-trans/ --n_items 8157 --train_batch_size 100 --arch_type MultiVAE --runs 5 --gpu 0 > output/trans_ml_mult_vae.out

python demo.py train --dataset_name ml-latest-small --processed_dir gowalla_trans/ --n_items 40981 --train_batch_size 512 --arch_type MultiVAE --runs 10 --gpu 0 > output/trans_gowalla_mult_vae.out

python demo.py train --dataset_name ml-latest-small --processed_dir yelp2018_trans/ --n_items 38048 --train_batch_size 512 --arch_type MultiVAE --runs 10 --gpu 0 > output/trans_yelp2018_mult_vae.out

python demo.py train --dataset_name ml-latest-small --processed_dir yelp2018_trans/ --n_items 38048 --train_batch_size 512 --arch_type MultiVAE --runs 10 --gpu 0 > output/trans_yelp2018_mult_vae.out




python demo.py train --dataset_name amazon-book --processed_dir amazon_book_trans/ --n_items 91597 --train_batch_size 512 --arch_type MultiVAE --runs 1 --gpu 0 > output/trans_amazon_mult_vae.out


python demo.py train --dataset_name amazon-book --processed_dir amazon_book/ --n_items 91599 --train_batch_size 512 --arch_type MultiVAE --runs 1 --gpu 1 > output/ind_amazon_mult_vae.out

python demo.py train --dataset_name ml_1m --processed_dir ml_1m/ --n_items 91599 --train_batch_size 512 --arch_type MultiVAE --runs 1 --gpu 1 > output/ind_ml1m_mult_vae.out
