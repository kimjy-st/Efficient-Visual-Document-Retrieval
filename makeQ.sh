# python3 makeQ.py \
#     --img_dir /home/yjjeon/Efficient_VDR/data/vidore_test/arxivqa_test_subsampled/image/test \
#     --out_json ProxyQ/arxivqa_test_subsampled.json \
#     --nq 50 --save_every 20


# python3 makeQ.py \
#     --img_dir /home/yjjeon/Efficient_VDR/data/vidore_test/docvqa_test_subsampled/image/test \
#     --out_json ProxyQ/docvqa_test_subsampled.json \
#     --nq 50 --save_every 20

# python3 makeQ.py \
#     --img_dir /home/yjjeon/Efficient_VDR/data/vidore_test/infovqa_test_subsampled/image/test \
#     --out_json ProxyQ/infovqa_test_subsampled.json\
#     --nq 50 --save_every 20

python3 makeQ.py \
    --img_dir /home/yjjeon/Efficient_VDR/data/vidore_test/tabfquad_test_subsampled/image/test \
    --out_json ProxyQ/tabfquad_test_subsampled.json \
    --nq 50 --save_every 20

python3 makeQ.py \
    --img_dir /home/yjjeon/Efficient_VDR/data/vidore_test/tatdqa_test/image/test \
    --out_json ProxyQ/tatdqa_test.json \
    --nq 50 --save_every 20

python3 makeQ.py \
    --img_dir /home/yjjeon/Efficient_VDR/data/vidore_test/shiftproject_test/image \
    --out_json ProxyQ/shiftproject_test.json \
    --nq 50 --save_every 20

python3 makeQ.py \
    --img_dir /home/yjjeon/Efficient_VDR/data/vidore_test/syntheticDocQA_artificial_intelligence_test/image \
    --out_json ProxyQ/syntheticDocQA_artificial_intelligence_test.json \
    --nq 50 --save_every 20

python3 makeQ.py \
    --img_dir /home/yjjeon/Efficient_VDR/data/vidore_test/syntheticDocQA_energy_test/image \
    --out_json ProxyQ/syntheticDocQA_energy_test.json \
    --nq 50 --save_every 20

python3 makeQ.py \
    --img_dir /home/yjjeon/Efficient_VDR/data/vidore_test/syntheticDocQA_government_reports_test/image \
    --out_json ProxyQ/syntheticDocQA_government_reports_test.json \
    --nq 50 --save_every 20

python3 makeQ.py \
    --img_dir /home/yjjeon/Efficient_VDR/data/vidore_test/syntheticDocQA_healthcare_industry_test/image \
    --out_json ProxyQ/syntheticDocQA_healthcare_industry_test.json \
    --nq 50 --save_every 20
