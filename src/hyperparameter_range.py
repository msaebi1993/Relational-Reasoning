"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Hyperparameter range specification.
"""

hp_range = {
    "beta": [ .02, .05],
    "emb_dropout_rate": [ .1, .3],
    #"ff_dropout_rate": [ 0.05, .1, .2],
    "action_dropout_rate": [0.1,0.5,0.9],
    "bandwidth": [128, 200, 300],
    "relation_only": [True, False]
}
