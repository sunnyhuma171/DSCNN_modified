"""
The code is modified from https://github.com/ryanzhumich/dscnn
"""
import cPickle as pkl
import sys
import argparse
from train import train_model
import hdf

if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    def str2bool(v):
        return v.lower() in ('true','1')
    parser.register('type','bool',str2bool)

    parser.add_argument('-dataset', default='hdf')
    parser.add_argument('-We',    default='wikihdf', help='Word Embedding')
    parser.add_argument('-bidir',   type='bool', default=False, help='Bidirectional RNN')
    parser.add_argument('-rnnshare',type='bool', default=False, help='')
    parser.add_argument('-rnnlayer',type=int, default=1)
    parser.add_argument('-deep',    type=int, default=0)
    parser.add_argument('-encoder', default='cnnlstm')
    parser.add_argument('-optim',   default='adadelta')
    parser.add_argument('-dropout_penul', type=float, default=0.5)
    parser.add_argument('-decay_c', type=float, default=0)
    parser.add_argument('-pool_type', default='max')
    parser.add_argument('-filter_hs', default='345')
    parser.add_argument('-combine',type='bool', default=False, help='')
    parser.add_argument('-feature_maps', type=int, default=100)
    parser.add_argument('-rm_unk',type='bool', default=False, help='')
    parser.add_argument('-validportion', type=float, default=0.1)
    parser.add_argument('-batchsize', type=int, default=28) 
    parser.add_argument('-init',  default='uniform')
    parser.add_argument('-salstm',type='bool', default=False, help='')
    args = vars(parser.parse_args())
    print args

    dataset = args['dataset']

    filter_hs = [int(h) for h in list(args['filter_hs'])]
    pad = max(filter_hs) - 1

    datasets={}
    if dataset == 'hdf':
        data_path = r'D:\python-nlpir-master\DSCNN_wiki_hdf\data\hdf\hdf.p'
        x = pkl.load(open(data_path,"rb"))
        revs,W,W2,W3,word_idx_map,vocab = x[0],x[1],x[2],x[3],x[4],x[5]
        
        Ws = []
        if 'wiki' in args['We']:
            print 'loading wiki...'
            We = W
            Ws.append(We)
   
        if 'random' in args['We']:
            print 'loading random...'
            We = W2
            Ws.append(We)

        if 'hdf' in args['We']:
            print 'loading hdf...'
            We = W3
            Ws.append(We)

        vocab_size,k = Ws[0].shape
        vocab_size -=1
  
        maxlen = 112

        datasets['hdf'] = (lambda:hdf.load_data(revs,word_idx_map,valid_portion=args['validportion']), lambda x,y:hdf.prepare_data(x,y,maxlen=maxlen,pad=pad))

        data_loader = datasets[dataset]
        perf = train_model(dim_proj = k,
                           decay_c = args['decay_c'],
                           n_words = vocab_size,
                           dataset = dataset,
                           W = Ws,
                           encoder = args['encoder'],
                           batch_size = args['batchsize'],
                           deep = args['deep'],
                           rnnlayer = args['rnnlayer'],
                           filter_hs = filter_hs,
                           maxlen=maxlen+2*pad,
                           dropout_penul = args['dropout_penul'],
                           pool_type = args['pool_type'],
                           combine = args['combine'],
                           feature_maps = args['feature_maps'],
                           init = args['init'],
                           data_loader = data_loader,
                           optimizer = args['optim'],
                           rnnshare = args['rnnshare'],
                           bidir = args['bidir'],
                           salstm = args['salstm'],
                           )
        p = "%.2f%%\t%.2f%%\t%.2f%%\n" % perf

    print p
