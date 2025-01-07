# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
from .uniform import UniformObserver
from paddle import _legacy_C_ops as _C_ops
from paddle.quantization.factory import ObserverFactory



class AbsmaxTokenwiseObserver(ObserverFactory):
    r"""
    It dynamically collects the maximum absolute value of the target tensor.
    Args:
        quant_bits(int, optional): Number of bits to represent an quantized integer in binary.
    """

    def __init__(self, quant_bits=8):
        super(AbsmaxTokenwiseObserver, self).__init__(quant_bits=quant_bits)

    def _get_class(self):
        return AbsmaxTokenwiseObserverLayer


class AbsmaxTokenwiseObserverLayer(UniformObserver):
    """
        为适配推理过程，校准时存储per tensor的最大值，
        fake quant时动态统计每个token的最大值，进行动态计算
        
        Args:
            layer (nn.Module): 需要观测的张量所在的层。
            quant_bits (int, optional): 激活量化的位数，默认为8位。
        
        Returns:
            AbsmaxTokenwiseObserverLayer: 返回一个 AbsmaxTokenwiseObserverLayer 对象。
        
        """
    def __init__(
            self,
            layer,
            quant_bits=8,):
        super(AbsmaxTokenwiseObserverLayer, self).__init__(quant_bits=quant_bits)
        self._quant_bits = quant_bits
        self._layer = layer
        self._scale = None
        self._zero_point = None
        self._min = None
        self._max = None #paddle.to_tensor(1e-7)
        self.step = 0
        self.fake_quant = False
        self.observer_enabled = True

    def forward(self, inputs):
        """ Calculate forward pass.
        """
        if self.observer_enabled:
            self._min, self._max = self.cal_min_max(inputs)
            return inputs
        if self.fake_quant:
            scales = paddle.max(paddle.abs(inputs), axis=-1, keepdim=True)
            scales = paddle.where(scales == paddle.to_tensor(0, dtype="bfloat16"),
                                      paddle.to_tensor(1e-8, dtype="bfloat16"), scales)
            bnt = (1 << (self._quant_bits - 1)) - 1
            quant_tensor = paddle.clip(paddle.round(inputs / scales * bnt), -bnt, bnt)
            dequant_tensor = quant_tensor * scales / bnt
            return dequant_tensor


    def cal_min_max(self, inputs):
        """ Compute min and max values.
        """
        abs_max_val = paddle.max(paddle.abs(inputs))
        if self._max is not None:
            abs_max_val = paddle.maximum(abs_max_val, self._max.cast(inputs.dtype))
        return 0, abs_max_val
        
    def cal_thresholds(self):
        """ Compute thresholds for MAX function.
        """
        if self._scale is not None:
            self._zero_point = 0
            return 
        self._scale, self._zero_point = self.cal_scales_zero_points()

    def min_value(self) -> float:
        """ Return min value
        """
        return self._min

    def max_value(self) -> float:
        """ Return max value
        """
        return self._max

    def bit_length(self):
        """ Return the bit length of quantized data.
        """
        return self._quant_bits

    def quant_axis(self):
        """ Return quantization axis.
        """
        return -1

    def scales(self):
        """ Return output scales.
        """
        self.cal_thresholds()
        return self._scale

    def zero_points(self):
        """ Return output zero points.
        """
        self.cal_thresholds()
        return self._zero_point