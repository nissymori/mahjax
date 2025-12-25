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

from mahjax._src.struct import dataclass


@dataclass
class Action:
    # Discard from hand: 0~33
    # Closed/Added Kan: 34~67
    TSUMOGIRI: int = 68
    RIICHI: int = 69
    TSUMO: int = 70
    RON: int = 71
    PON: int = 72
    OPEN_KAN: int = 73
    CHI_L: int = 74  # [4]56
    CHI_M: int = 75  # 4[5]6
    CHI_R: int = 76  # 45[6]
    PASS: int = 77
    DUMMY: int = 78  # For sharing information after round.
    NUM_ACTION: int = 79

    @staticmethod
    def is_selfkan(action: int) -> bool:
        return (34 <= action) & (action < 68)
