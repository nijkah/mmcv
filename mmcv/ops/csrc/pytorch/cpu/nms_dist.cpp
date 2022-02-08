// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

Tensor nms_dist_cpu(Tensor boxes, Tensor scores, float dist_threshold, int offset) {
  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }
  auto x1_t = boxes.select(1, 0).contiguous();
  auto y1_t = boxes.select(1, 1).contiguous();

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto nboxes = boxes.size(0);
  Tensor select_t = at::ones({nboxes}, boxes.options().dtype(at::kBool));

  auto select = select_t.data_ptr<bool>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<float>();
  auto y1 = y1_t.data_ptr<float>();

  for (int64_t _i = 0; _i < nboxes; _i++) {
    if (select[_i] == false) continue;
    auto i = order[_i];
    auto ix1 = x1[i];
    auto iy1 = y1[i];

    for (int64_t _j = _i + 1; _j < nboxes; _j++) {
      if (select[_j] == false) continue;
      auto j = order[_j];
      auto xdist = std::pow(ix1-x1[j], 2);
      auto ydist = std::pow(iy1-y1[j], 2);
      auto dist = std::sqrt(xdist+ydist);

      if (dist < dist_threshold) select[_j] = false;
    }
  }
  return order_t.masked_select(select_t);
}

Tensor nms_dist_impl(Tensor boxes, Tensor scores, float dist_threshold, int offset);
REGISTER_DEVICE_IMPL(nms_dist_impl, CPU, nms_dist_cpu);

