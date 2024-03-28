# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
# Copyright (c) 2024 Timekettle Inc. (authors: Sirui Li)
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

import re

import logging
import coloredlogs


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def setup_logger(name, log_file, level='INFO', fmt='%(asctime)s %(funcName)s %(levelname)s %(message)s'):
    """To setup as many loggers as you want"""

    formatter = logging.Formatter(fmt)

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)

    logger.addHandler(handler)

    # Create console handler with the same formatter and add it to the logger
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    level_styles = {
        'debug': {'color': 'magenta'},
        'info': {'color': 'green'},
        'warning': {'color': 'yellow'}
    }
    coloredlogs.install(level=level, level_styles=level_styles,
                        fmt='%(asctime)s %(funcName)s %(levelname)s %(message)s', logger=logger)

    return logger
