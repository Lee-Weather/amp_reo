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

from .x1.x1_amp_config import (X1AMPCfg, X1AMPCfgPPO,
                               X1AMPFinalCfg, X1AMPFinalCfgPPO)
from .x1.x1_amp_env import X1AMPEnv, X1AMPFinalEnv

# x1_amp is the consolidated exp2.12 behaviour (one config + one env file).
# The original exp1.0 baseline remains runnable as x1_amp_base and is still the
# base class of every x1_amp_exp* task.
_x1_amp_base_ppo = X1AMPCfgPPO()
_x1_amp_base_ppo.runner.experiment_name = "x1_amp_base"
task_registry.register("x1_amp", X1AMPFinalEnv, X1AMPFinalCfg(), X1AMPFinalCfgPPO())
task_registry.register("x1_amp_base", X1AMPEnv, X1AMPCfg(), _x1_amp_base_ppo)

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

from .x1.x1_amp_exp19_config import X1AMPExp19Cfg, X1AMPExp19CfgPPO

task_registry.register("x1_amp_exp19", X1AMPExp18Env, X1AMPExp19Cfg(), X1AMPExp19CfgPPO())

from .x1.x1_amp_exp30_config import X1AMPExp30Cfg, X1AMPExp30CfgPPO
from .x1.x1_amp_exp30_env import X1AMPExp30Env

task_registry.register("x1_amp_exp30", X1AMPExp30Env, X1AMPExp30Cfg(), X1AMPExp30CfgPPO())

from .x1.x1_amp_exp31_config import X1AMPExp31Cfg, X1AMPExp31CfgPPO
from .x1.x1_amp_exp31_env import X1AMPExp31Env

task_registry.register("x1_amp_exp31", X1AMPExp31Env, X1AMPExp31Cfg(), X1AMPExp31CfgPPO())

from .x1.x1_amp_exp32_config import X1AMPExp32Cfg, X1AMPExp32CfgPPO
from .x1.x1_amp_exp32_env import X1AMPExp32Env

task_registry.register("x1_amp_exp32", X1AMPExp32Env, X1AMPExp32Cfg(), X1AMPExp32CfgPPO())

from .x1.x1_amp_exp33_config import X1AMPExp33Cfg, X1AMPExp33CfgPPO
from .x1.x1_amp_exp33_env import X1AMPExp33Env

task_registry.register("x1_amp_exp33", X1AMPExp33Env, X1AMPExp33Cfg(), X1AMPExp33CfgPPO())

from .x1.x1_amp_exp34_config import X1AMPExp34Cfg, X1AMPExp34CfgPPO
from .x1.x1_amp_exp34_env import X1AMPExp34Env

task_registry.register("x1_amp_exp34", X1AMPExp34Env, X1AMPExp34Cfg(), X1AMPExp34CfgPPO())

from .x1.x1_amp_exp35_config import X1AMPExp35Cfg, X1AMPExp35CfgPPO
from .x1.x1_amp_exp35_env import X1AMPExp35Env

task_registry.register("x1_amp_exp35", X1AMPExp35Env, X1AMPExp35Cfg(), X1AMPExp35CfgPPO())

from .x1.x1_amp_exp36_config import X1AMPExp36Cfg, X1AMPExp36CfgPPO
from .x1.x1_amp_exp36_env import X1AMPExp36Env

task_registry.register("x1_amp_exp36", X1AMPExp36Env, X1AMPExp36Cfg(), X1AMPExp36CfgPPO())

from .x1.x1_amp_exp110_config import X1AMPExp110Cfg, X1AMPExp110CfgPPO

task_registry.register("x1_amp_exp110", X1AMPExp18Env, X1AMPExp110Cfg(), X1AMPExp110CfgPPO())

from .x1.x1_amp_exp111_config import X1AMPExp111Cfg, X1AMPExp111CfgPPO

task_registry.register("x1_amp_exp111", X1AMPExp18Env, X1AMPExp111Cfg(), X1AMPExp111CfgPPO())

from .x1.x1_amp_exp112_config import X1AMPExp112Cfg, X1AMPExp112CfgPPO
from .x1.x1_amp_exp112_env import X1AMPExp112Env

task_registry.register("x1_amp_exp112", X1AMPExp112Env, X1AMPExp112Cfg(), X1AMPExp112CfgPPO())

from .x1.x1_amp_exp113_config import X1AMPExp113Cfg, X1AMPExp113CfgPPO
from .x1.x1_amp_exp113_env import X1AMPExp113Env

task_registry.register("x1_amp_exp113", X1AMPExp113Env, X1AMPExp113Cfg(), X1AMPExp113CfgPPO())

from .x1.x1_amp_exp113t_config import X1AMPExp113tCfg, X1AMPExp113tCfgPPO
from .x1.x1_amp_exp113t_env import X1AMPExp113tEnv

task_registry.register("x1_amp_exp113t", X1AMPExp113tEnv, X1AMPExp113tCfg(), X1AMPExp113tCfgPPO())

from .x1.x1_amp_exp114_config import X1AMPExp114Cfg, X1AMPExp114CfgPPO
from .x1.x1_amp_exp114_env import X1AMPExp114Env

task_registry.register("x1_amp_exp114", X1AMPExp114Env, X1AMPExp114Cfg(), X1AMPExp114CfgPPO())

from .x1.x1_amp_exp114t_config import X1AMPExp114tCfg, X1AMPExp114tCfgPPO
from .x1.x1_amp_exp114t_env import X1AMPExp114tEnv

task_registry.register("x1_amp_exp114t", X1AMPExp114tEnv, X1AMPExp114tCfg(), X1AMPExp114tCfgPPO())

from .x1.x1_amp_exp115_config import X1AMPExp115Cfg, X1AMPExp115CfgPPO
from .x1.x1_amp_exp115t_config import X1AMPExp115tCfg, X1AMPExp115tCfgPPO
from .x1.x1_amp_exp115_env import X1AMPExp115Env

task_registry.register("x1_amp_exp115", X1AMPExp115Env, X1AMPExp115Cfg(), X1AMPExp115CfgPPO())
task_registry.register("x1_amp_exp115t", X1AMPExp115Env, X1AMPExp115tCfg(), X1AMPExp115tCfgPPO())

from .x1.x1_amp_exp116_config import X1AMPExp116Cfg, X1AMPExp116CfgPPO
from .x1.x1_amp_exp116_env import X1AMPExp116Env

task_registry.register("x1_amp_exp116", X1AMPExp116Env, X1AMPExp116Cfg(), X1AMPExp116CfgPPO())

from .x1.x1_amp_exp117_config import X1AMPExp117Cfg, X1AMPExp117CfgPPO
from .x1.x1_amp_exp117_env import X1AMPExp117Env

task_registry.register("x1_amp_exp117", X1AMPExp117Env, X1AMPExp117Cfg(), X1AMPExp117CfgPPO())

from .x1.x1_amp_exp118_config import X1AMPExp118Cfg, X1AMPExp118CfgPPO

task_registry.register("x1_amp_exp118", X1AMPExp117Env, X1AMPExp118Cfg(), X1AMPExp118CfgPPO())

from .x1.x1_amp_exp20_config import X1AMPExp20Cfg,X1AMPExp20CfgPPO
from .x1.x1_amp_exp20_env import X1AMPExp20Env

task_registry.register('x1_amp_exp20',X1AMPExp20Env,X1AMPExp20Cfg(),X1AMPExp20CfgPPO())

from .x1.x1_amp_exp21_config import X1AMPExp21Cfg,X1AMPExp21CfgPPO
from .x1.x1_amp_exp21_env import X1AMPExp21Env
task_registry.register('x1_amp_exp21',X1AMPExp21Env,X1AMPExp21Cfg(),X1AMPExp21CfgPPO())

from .x1.x1_amp_exp22_config import X1AMPExp22Cfg,X1AMPExp22CfgPPO
from .x1.x1_amp_exp22_env import X1AMPExp22Env
task_registry.register('x1_amp_exp22',X1AMPExp22Env,X1AMPExp22Cfg(),X1AMPExp22CfgPPO())

from .x1.x1_amp_exp23_config import X1AMPExp23Cfg,X1AMPExp23CfgPPO
from .x1.x1_amp_exp23_env import X1AMPExp23Env
task_registry.register('x1_amp_exp23',X1AMPExp23Env,X1AMPExp23Cfg(),X1AMPExp23CfgPPO())

from .x1.x1_amp_exp24_config import X1AMPExp24Cfg,X1AMPExp24CfgPPO
from .x1.x1_amp_exp24_env import X1AMPExp24Env
task_registry.register('x1_amp_exp24',X1AMPExp24Env,X1AMPExp24Cfg(),X1AMPExp24CfgPPO())

from .x1.x1_amp_exp25_config import X1AMPExp25Cfg,X1AMPExp25CfgPPO
from .x1.x1_amp_exp25_env import X1AMPExp25Env
task_registry.register('x1_amp_exp25',X1AMPExp25Env,X1AMPExp25Cfg(),X1AMPExp25CfgPPO())

from .x1.x1_amp_exp26_config import X1AMPExp26Cfg,X1AMPExp26CfgPPO
from .x1.x1_amp_exp26_env import X1AMPExp26Env
task_registry.register('x1_amp_exp26',X1AMPExp26Env,X1AMPExp26Cfg(),X1AMPExp26CfgPPO())

from .x1.x1_amp_exp27_config import X1AMPExp27Cfg,X1AMPExp27CfgPPO
from .x1.x1_amp_exp27_env import X1AMPExp27Env
task_registry.register('x1_amp_exp27',X1AMPExp27Env,X1AMPExp27Cfg(),X1AMPExp27CfgPPO())

from .x1.x1_amp_exp28_config import X1AMPExp28Cfg,X1AMPExp28CfgPPO
from .x1.x1_amp_exp28_env import X1AMPExp28Env
task_registry.register('x1_amp_exp28',X1AMPExp28Env,X1AMPExp28Cfg(),X1AMPExp28CfgPPO())

from .x1.x1_amp_exp29_config import X1AMPExp29Cfg,X1AMPExp29CfgPPO
from .x1.x1_amp_exp29_env import X1AMPExp29Env
task_registry.register('x1_amp_exp29',X1AMPExp29Env,X1AMPExp29Cfg(),X1AMPExp29CfgPPO())

from .x1.x1_amp_exp210_config import X1AMPExp210Cfg,X1AMPExp210CfgPPO
from .x1.x1_amp_exp210_env import X1AMPExp210Env

task_registry.register('x1_amp_exp210',X1AMPExp210Env,X1AMPExp210Cfg(),X1AMPExp210CfgPPO())

from .x1.x1_amp_exp211_config import X1AMPExp211Cfg,X1AMPExp211CfgPPO
from .x1.x1_amp_exp211_env import X1AMPExp211Env

task_registry.register('x1_amp_exp211',X1AMPExp211Env,X1AMPExp211Cfg(),X1AMPExp211CfgPPO())

from .x1.x1_amp_exp212_config import X1AMPExp212Cfg,X1AMPExp212CfgPPO
from .x1.x1_amp_exp212_env import X1AMPExp212Env

task_registry.register('x1_amp_exp212',X1AMPExp212Env,X1AMPExp212Cfg(),X1AMPExp212CfgPPO())
