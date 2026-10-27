# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.


from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .x1.x1_dh_stand_config  import X1DHStandCfg, X1DHStandCfgPPO

from .x1.x1_dh_stand_env import X1DHStandEnv

from humanoid.utils.task_registry import task_registry

task_registry.register( "x1_dh_stand", X1DHStandEnv, X1DHStandCfg(), X1DHStandCfgPPO() )

from .x1.x1_amp_config import X1AMPCfg, X1AMPCfgPPO
from .x1.x1_amp_env import X1AMPEnv

task_registry.register("x1_amp", X1AMPEnv, X1AMPCfg(), X1AMPCfgPPO())

from .x1.x1_amp_exp11_config import X1AMPExp11Cfg, X1AMPExp11CfgPPO
from .x1.x1_amp_exp11_env import X1AMPExp11Env

task_registry.register("x1_amp_exp11", X1AMPExp11Env, X1AMPExp11Cfg(), X1AMPExp11CfgPPO())

from .x1.x1_amp_exp3_config import X1AMPExp3Cfg, X1AMPExp3CfgPPO
from .x1.x1_amp_exp3_env import X1AMPExp3Env

task_registry.register("x1_amp_exp3", X1AMPExp3Env, X1AMPExp3Cfg(), X1AMPExp3CfgPPO())

from .x1.x1_amp_exp14_config import X1AMPExp14Cfg, X1AMPExp14CfgPPO
from .x1.x1_amp_exp14_env import X1AMPExp14Env

task_registry.register("x1_amp_exp14", X1AMPExp14Env, X1AMPExp14Cfg(), X1AMPExp14CfgPPO())

from .x1.x1_amp_exp15_config import X1AMPExp15Cfg, X1AMPExp15CfgPPO
from .x1.x1_amp_exp15_env import X1AMPExp15Env

task_registry.register("x1_amp_exp15", X1AMPExp15Env, X1AMPExp15Cfg(), X1AMPExp15CfgPPO())

from .x1.x1_amp_exp16_config import X1AMPExp16Cfg, X1AMPExp16CfgPPO
from .x1.x1_amp_exp16_env import X1AMPExp16Env

task_registry.register("x1_amp_exp16", X1AMPExp16Env, X1AMPExp16Cfg(), X1AMPExp16CfgPPO())

from .x1.x1_amp_exp17_config import X1AMPExp17Cfg, X1AMPExp17CfgPPO
from .x1.x1_amp_exp17_env import X1AMPExp17Env

task_registry.register("x1_amp_exp17", X1AMPExp17Env, X1AMPExp17Cfg(), X1AMPExp17CfgPPO())

from .x1.x1_amp_exp18_config import X1AMPExp18Cfg, X1AMPExp18CfgPPO
from .x1.x1_amp_exp18_env import X1AMPExp18Env

task_registry.register("x1_amp_exp18", X1AMPExp18Env, X1AMPExp18Cfg(), X1AMPExp18CfgPPO())
