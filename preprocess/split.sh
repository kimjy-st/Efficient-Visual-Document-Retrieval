

python3 split_data.py \
    --in_npz '/home/hyshin/Project/EVDR/SIGIR/features2/colqwen/docvqa_test_subsampled_dump_new.npz' \
    --out_dir '../data/split_features/colqwen'


python3 split_data.py \
    --in_npz '/home/hyshin/Project/EVDR/SIGIR/features2/colqwen/arxivqa_test_subsampled_dump_new.npz' \
    --out_dir '../data/split_features/colqwen'

python3 split_data.py \
    --in_npz '/home/hyshin/Project/EVDR/SIGIR/features2/colqwen/tatdqa_test_dump_new.npz' \
    --out_dir '../data/split_features/colqwen'


python3 split_data.py \
    --in_npz '/home/hyshin/Project/EVDR/SIGIR/features2/colqwen/infovqa_test_subsampled_dump_new.npz' \
    --out_dir '../data/split_features/colqwen'

python3 split_data.py \
    --in_npz '/home/hyshin/Project/EVDR/SIGIR/features2/colqwen/tabfquad_test_subsampled_dump_new.npz' \
    --out_dir '../data/split_features/colqwen'
