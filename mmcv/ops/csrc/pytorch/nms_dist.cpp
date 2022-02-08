// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

Tensor nms_dist_impl(Tensor boxes, Tensor scores, float dist_threshold, int offset) {
  return DISPATCH_DEVICE_IMPL(nms_dist_impl, boxes, scores, dist_threshold, offset);
}

Tensor nms_dist(Tensor boxes, Tensor scores, float dist_threshold, int offset) {
  return nms_dist_impl(boxes, scores, dist_threshold, offset);
}


