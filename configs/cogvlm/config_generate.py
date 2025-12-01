import json

# 原始模板文件
src = "/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_s0_rank_20_cogvlm.json"

# 输出文件模板
out_template = "/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_s{start}_rank_20_cogvlm.json"

# train_csv_path 模板
csv_template = "/m2v_intern/precomputed_latents/250617_laion5b_100m_filter_30m/rater_csv/250617_laion5b_100m_trainset_filter_filter_30mTrain_withRating0801_sorted_start_{start}_portion_20.csv"

# local_out_dir_path 模板
outdir_template = "/m2v_intern/zhouyang15/codes/STAR-T2I/outdir/30m_d16_256-0904_rank_s{start}_p20_cogvlm"

# 要生成的 start 列表
start_list = [0, 10, 20, 30, 40, 50, 60, 70, 80]

for start in start_list:
    # 读原始 JSON 模板
    with open(src, "r") as f:
        cfg = json.load(f)

    # 修改字段
    cfg["data_args"]["train_csv_path"] = csv_template.format(start=start)
    cfg["local_out_dir_path"] = outdir_template.format(start=start)

    # 保存新文件
    out_path = out_template.format(start=start)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"已生成: {out_path}")