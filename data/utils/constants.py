"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
TOKENS_PER_CELL = 7
RXYZ = [0, 1, 2]
RPY = [3, 4]
RBM = [5, 6, 7]
RVOL = [8]
RBBOX = [9, 10, 11, 12, 13, 14]

T_HASH = 4
T_FLOAT = 3
T_WORD = 4 * TOKENS_PER_CELL
R_HASH = 2
R_WORD = 5 * TOKENS_PER_CELL
REFOBJ_FLOAT = 15
