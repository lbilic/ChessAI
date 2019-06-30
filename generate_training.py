#!root/anaconda3/envs/ORI/bin/python
import os
import numpy as np
import chess.pgn
from state import State
import h5py

def get_dataset(num_samples=None):
    #X, Y = [], []
    idx = 0
    with h5py.File(os.path.join('processed', 'dataset10M.h5'), 'w') as f:
        d1 = f.create_dataset('array_1',(10000000, 5, 8, 8))
        d2 = f.create_dataset('array_2', (10000000, 1))
        gn = 0
        values = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
        for fn in os.listdir(os.path.join("dataset", "PGN")):
            pgn = open(os.path.join("dataset", "PGN", fn), encoding='utf8', errors='ignore')
            while 1:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                res = game.headers['Result']
                if res not in values:
                    continue
                value = values[res]
                board = game.board()
                for i, move in enumerate(game.mainline_moves()):
                    board.push(move)
                    ser = State(board).serialize()
                    d1[idx] = ser
                    d2[idx] = value
                    idx += 1
                #X.append(ser)
                #Y.append(value)
                print("parsing game %d, got %d examples" % (gn, idx))
                if(idx == 10000000):
                    break
            #if num_samples is not None and len(X) > num_samples:
                #return X,Y
                gn += 1
    #X = np.array(X)
    #Y = np.array(Y)
    #return X, Y

if __name__ == '__main__':
    get_dataset()
    #X, Y = get_dataset(10000000)
    #np.savez("processed/dataset.npz", X, Y)
