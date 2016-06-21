#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import Bio.PDB
import Bio.PDB.PDBParser
import argparse
import matplotlib.pyplot as plt


class BioInformatics:
    def __init__(self, pdb_name, chain_name):
        self.pdb_name = pdb_name
        self.chain_name = chain_name

    def get_positions(self):
        # pdbファイルのパース
        pdb_parser = Bio.PDB.PDBParser()
        structure = pdb_parser.get_structure(id='X', file='{0}'.format(self.pdb_name))
        # 座標配列の生成
        positions = np.zeros((1, 3), dtype=np.float32)
        if self.chain_name:
            # 指定されたchainのCA原子すべての座標を取得
            for model in structure.get_list():
                for chain in model.get_list():
                    # 指定したchain名以外は配列に格納しない
                    if chain.get_id() != self.chain_name:
                        continue
                    for residue in chain.get_list():
                        for atom in residue.get_list():
                            # 原子の名前が CA とであるものを選ぶ
                            if atom.get_name() == 'CA':
                                positions = np.vstack((positions, atom.get_coord()))
        else:
            # CA原子すべての座標を取得
            for model in structure.get_list():
                for chain in model.get_list():
                    for residue in chain.get_list():
                        for atom in residue.get_list():
                            # 原子の名前が CA とであるものを選ぶ
                            if atom.get_name() == 'CA':
                                positions = np.vstack((positions, atom.get_coord()))
        # 初期化のために用いた1列目以外を使用
        positions = positions[1:]
        return positions

    def get_distances(self, positions):
        # 二つのベクトルの差を全て計算
        # pythonではfor文を使うと計算に時間がかかるのでnumpy配列を使用
        diff = positions[:, np.newaxis] - positions
        # ベクトルの差のノルムをとって原子間距離を求め、配列に格納
        diff_flatten = np.ravel(np.linalg.norm(diff, axis=2))
        # 同一の原子についても距離を計算してしまっているので差が0のものを削除
        diff_flatten = diff_flatten[np.where(diff_flatten != 0)]
        return diff_flatten

    def get_hist(self):
        positions = self.get_positions()
        diff_flatten = self.get_distances(positions)
        # 距離の最大値を取得
        max_distance = np.ceil(np.max(diff_flatten))
        # ヒストグラムを作成、ビンは0からmax_distanceを1ずつ区切る。
        hist, bin_edges = np.histogram(diff_flatten, bins=np.arange(0, max_distance + 1))
        # 距離を求める際に、本来は組み合わせのとことろを順列で計算しているので2で割る。
        hist /= 2
        return hist, bin_edges


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-f1', '--file1', help='pdb file name', required=True)
    p.add_argument('-c1', '--chain1', help='chain name if you want to focus on specific chain')
    p.add_argument('-f2', '--file2', help='pdb file name if you want to compare two files')
    p.add_argument('-c2', '--chain2', help='chain name if you want to focus on specific chain')
    args = p.parse_args()

    p1 = BioInformatics(args.file1, args.chain1)
    hist1, bin_edges1 = p1.get_hist()
    plt.bar(bin_edges1[:-1], hist1, width=1, alpha=0.3, color='r', label='{0}'.format(args.file1))
    if args.file2:
        p2 = BioInformatics(args.file2, args.chain2)
        hist2, bin_edges2 = p2.get_hist()
        # グラフを重ねるとx軸の右端は自動的に二つのうち大きい値となる。
        plt.bar(bin_edges2[:-1], hist2, width=1, alpha=0.3, color='b', label='{0}'.format(args.file2))
    plt.legend(loc='upper right')
    plt.show()
