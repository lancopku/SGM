def model_opts(parser):

    parser.add_argument('-config', default='config.yaml', type=str,
                        help="config file")
    parser.add_argument('-gpus', default=[], nargs='+', type=int,
                        help="use CUDA on the listed devices.")
    parser.add_argument('-restore', default='./checkpoints/sgmge.pt', type=str,
                        help="restore checkpoint")
    parser.add_argument('-seed', default=1234, type=int, 
                        help="random seed")
    parser.add_argument('-model', default='seq2seq', type=str,
                        help="model selection")
    parser.add_argument('-mode', default='train', type=str,
                        help="mode selection")
    parser.add_argument('-module', default='seq2seq', type=str,
                        help="module selection")
    parser.add_argument('-log', default='', type=str,
                        help="log directory")
    parser.add_argument('-num_processes', type=int, default=4,
                        help="number of processes")
    parser.add_argument('-refF', default='', type=str,
                        help="reference file")
    parser.add_argument('-unk', action='store_true', 
                        help='replace unk')
    parser.add_argument('-char', action='store_true', 
                        help='char level decoding')
    parser.add_argument('-length_norm', action='store_true', 
                        help='replace unk')
    parser.add_argument('-pool_size', type=int, default=0, 
                        help="pool size of maxout layer")
    parser.add_argument('-scale', type=float, default=1, 
                        help="proportion of the training set")
    parser.add_argument('-max_split', type=int, default=0, 
                        help="max generator time steps for memory efficiency")
    parser.add_argument('-split_num', type=int, default=0, 
                        help="split number for splitres")
    parser.add_argument('-pretrain', default='', type=str, 
                        help="load pretrain encoder")
    parser.add_argument('-label_dict_file', default='./data/topic_sorted.json', type=str,
                        help="label_dict")


def convert_to_config(opt, config):
    opt = vars(opt)
    for key in opt:
        if key not in config:
            config[key] = opt[key]
