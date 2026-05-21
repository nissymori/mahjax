# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .struct import dataclass


@dataclass
class Action:
    # Discard from hand: 0~36
    # Closed/Added Kan: 37~70
    TSUMOGIRI: int = 71
    RIICHI: int = 72
    TSUMO: int = 73
    RON: int = 74
    PON: int = 75
    PON_RED: int = 76
    OPEN_KAN: int = 77
    CHI_L: int = 78  # [4]56
    CHI_L_RED: int = 79  # [4]5r6
    CHI_M: int = 80  # 4[5r]6
    CHI_M_RED: int = 81  # 4[5r]6
    CHI_R: int = 82  # 45[6]
    CHI_R_RED: int = 83  # 45[r6]
    PASS: int = 84
    KYUUSHU: int = 85
    DUMMY: int = 86  # For sharing information after round.
    NUM_ACTION: int = 87

    @staticmethod
    def is_selfkan(action: int) -> bool:
        return (37 <= action) & (action < 71)