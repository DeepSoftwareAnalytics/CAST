#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import torch
import os


class Config(object):
    debug = True
    # ------dataset type------------
    dataset_type = "TL_CodeSum"  # funcom tl_codesum
    model_type = "x_RvNN_serial"
    # -------------------data processing ---------------------
    approach = "our"  # our  astnn hybrid-DRL Rencos
    language = "java"
    if dataset_type == "Funcom":
        data_root_path = "../../Data/Funcom"
    elif "TL_CodeSum" in dataset_type:
        data_root_path = "../../Data/TL_CodeSum"
    else:
        raise RuntimeError('Unsupported dataset_type: %s' % dataset_type)
    dot_files_dir = os.path.join(data_root_path, "splitted_ast")
    correct_fid = os.path.join(data_root_path, "correct_fids")

   # --- length --------------
    asts_len = -1  # use all sliced tree
    code_max_len = 100
    sum_max_len = 12

    if dataset_type == "Funcom":
        dataset_path = "../Data/Funcom"
    elif "TL_CodeSum" in dataset_type:
        dataset_path = "../Data/TL_CodeSum"
    else:
        raise RuntimeError('Unsupported dataset_type: %s' % dataset_type)
    epochs = 2
    batch_size = 2
    gpu_id = 0
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    ast_vocab_size = 100
    summary_vocab_size = 100
    code_vocab_size = 100

    # ---------------model---------------
    dim = 100
    node_embedding_dim = dim  # AST
    summary_embedding_dim = dim
    code_embedding_dim = dim

    dim1 = 200
    RvNN_input_dim = dim1  # hidden dim
    aggregate_dim = dim1
    aggregate_dim = 512

    lr = 0.001
    weight_decay = 0
    model_save_path = os.path.join("./", "output_mt_" + model_type + "_" + dataset_type, "model")
    num_subprocesses = 4
    is_DDP = torch.cuda.device_count() > 1
    # ----------transformer-------------
    x_hidden_dim = 256
    enc_layers = 3
    dec_layers = 3

    enc_heads = 8
    dec_heads = 8
    enc_pf_dim = 512  # hidden dim of  MLP
    dec_pf_dim = 512
    enc_dropout = 0.1
    dec_dropout = 0.1
    feature_dim = dim
    dim_feedforward = dim
    nhead = 2  # 8
    nlayers = 6  # 6
    dropout = 0.5

    # ---------------val---------------
    eval_result_save_path = os.path.join("./", "output_" + model_type, "predict")

    #  -----------------debug-----------
    DEBUG = False
    # -----early stop----------
    patience = 20

    random_seed = 123
    # ---pooling-------
    node_combine = "max"  # max ,mean
    encoder_pooling = "max"  # max avg
    paper_report = True
    # ---------RvNN----------------
    is_weighted_RvNN = True
    is_avg_weighted_RvNN = True
    is_rebuild_tree = True
    activate_f = "relu"  # tanh
    is_full_ast = False
    device = None
    use_clip = False
    clip = 5
    # ------Code, Sum, Ast path--------
    data_dir = os.path.join(dataset_path, "code_sum")
    dataset_name = "java"
    data_workers = 0
    java_doc_extension = "original"
    code_extension = "subtoken"

    train_src_files = os.path.join(data_dir, "train/code.%s" % code_extension)
    train_ast_files = os.path.join(dataset_path, "ASTs/train/split_AST.pkl")
    train_rebuild_tree_files = os.path.join(dataset_path, "ASTs/train/rebuild_tree.pkl")
    train_tgt_files = os.path.join(data_dir, "train/javadoc.%s" % java_doc_extension)
    ast_original_token_wc = os.path.join(dataset_path, "ASTs/asts_word_count.pkl")

    dev_src_files = os.path.join(data_dir, "val/code.%s" % code_extension)
    dev_ast_files = os.path.join(dataset_path, "ASTs/val/split_AST.pkl")
    dev_rebuild_tree_files = os.path.join(dataset_path, "ASTs/val/rebuild_tree.pkl")
    dev_tgt_files = os.path.join(data_dir, "val/javadoc.%s" % java_doc_extension)

    test_src_files = os.path.join(data_dir, "test/code.%s" % code_extension)
    test_ast_files = os.path.join(dataset_path, "ASTs/test/split_AST.pkl")
    test_rebuild_tree_files = os.path.join(dataset_path, "ASTs/test/rebuild_tree.pkl")
    test_tgt_files = os.path.join(data_dir, "test/javadoc.%s" % java_doc_extension)

    pred_file = ""
    big_graph_path = os.path.join(dataset_path, "ASTs/fids.pkl")

    uncase = True
    use_src_word = True
    use_src_char = False
    use_ast_word = True
    use_ast_char = False
    use_tgt_word = True
    use_tgt_char = False
    fix_embeddings = False
    share_decoder_embeddings = True
    max_examples = -1
    d_k = 64
    d_v = 64
    d_ff = 2048
    src_pos_emb = False
    tgt_pos_emb = True
    max_relative_pos = [32]
    use_neg_dist = True
    copy_attn = True
    early_stop = 20
    warmup_steps = 2000
    optimizer = "adam"
    lr_decay = 0.99
    valid_metric = "bleu"
    checkpoint = True
    only_test = False
    train_src_tag_files = None
    train_src_tag = ["None"]
    use_code_type = False
    code_tag_type = "subtoken"
    pretrained = False
    max_characters_per_token = 30
    dropout_emb = 0.2
    use_all_enc_layers = False
    split_decoder = False
    coverage_attn = False
    reload_decoder_state = None
    layer_wise_attn = False
    attn_type = "general"  # [dot, general, mlp]
    force_copy = False
    sort_by_len = True
    test_batch_size = True
    warmup_epochs = 0
    print_one_target = False
    checkpoint = True
    # ------Code2Sum--------
    aim = "_"
    gin_config_file = "c2nl.gin"
    arg_idx = 0
    local_rank = 0
    use_asts = True
    init_weights = False
    use_ast_sub_token = False
    rm_duplication = False