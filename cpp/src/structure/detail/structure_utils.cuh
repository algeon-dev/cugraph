/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/graph.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/mem_frugal_partition.cuh>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>
#include <cugraph/utilities/thrust_wrappers/gather.hpp>
#include <cugraph/utilities/thrust_wrappers/scan.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/device_atomics.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>

namespace cugraph {

namespace detail {

template <typename edge_t, typename VertexIterator>
rmm::device_uvector<edge_t> compute_sparse_offsets(
  raft::handle_t const& handle,
  VertexIterator edgelist_major_first,
  VertexIterator edgelist_major_last,
  typename thrust::iterator_traits<VertexIterator>::value_type major_range_first,
  typename thrust::iterator_traits<VertexIterator>::value_type major_range_last,
  bool edgelist_major_sorted,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  using vertex_t = typename thrust::iterator_traits<VertexIterator>::value_type;

  auto offset_array_size = static_cast<size_t>(major_range_last - major_range_first) + 1;
  auto offsets =
    large_buffer_type
      ? large_buffer_manager::allocate_memory_buffer<edge_t>(offset_array_size, handle.get_stream())
      : rmm::device_uvector<edge_t>(offset_array_size, handle.get_stream());
  if (edgelist_major_sorted) {
    offsets.set_element_to_zero_async(0, handle.get_stream());
    thrust::upper_bound(handle.get_thrust_policy(),
                        edgelist_major_first,
                        edgelist_major_last,
                        thrust::make_counting_iterator(major_range_first),
                        thrust::make_counting_iterator(major_range_last),
                        offsets.begin() + 1);
  } else {
    cugraph::fill(handle.get_thrust_policy(), offsets.begin(), offsets.end(), edge_t{0});

    if (large_buffer_type) {
      auto num_edges =
        static_cast<size_t>(cuda::std::distance(edgelist_major_first, edgelist_major_last));
      auto max_edges_to_process_per_iteration = std::min(
        static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 20),
        num_edges);

      rmm::device_uvector<vertex_t> indices(max_edges_to_process_per_iteration,
                                            handle.get_stream());
      rmm::device_uvector<vertex_t> output_indices(max_edges_to_process_per_iteration,
                                                   handle.get_stream());
      rmm::device_uvector<edge_t> output_counts(max_edges_to_process_per_iteration,
                                                handle.get_stream());

      size_t num_edges_processed{0};
      while (num_edges_processed < num_edges) {
        auto num_edges_to_process =
          std::min(num_edges - num_edges_processed, max_edges_to_process_per_iteration);
        thrust::transform(
          handle.get_thrust_policy(),
          edgelist_major_first + num_edges_processed,
          edgelist_major_first + (num_edges_processed + num_edges_to_process),
          indices.begin(),
          cuda::proclaim_return_type<vertex_t>(
            [major_range_first] __device__(auto major) { return major - major_range_first; }));
        cugraph::sort(
          handle.get_thrust_policy(), indices.begin(), indices.begin() + num_edges_to_process);
        auto it          = thrust::reduce_by_key(handle.get_thrust_policy(),
                                        indices.begin(),
                                        indices.begin() + num_edges_to_process,
                                        cuda::make_constant_iterator(edge_t{1}),
                                        output_indices.begin(),
                                        output_counts.begin());
        auto input_first = thrust::make_zip_iterator(output_indices.begin(), output_counts.begin());
        thrust::for_each(
          handle.get_thrust_policy(),
          input_first,
          input_first + cuda::std::distance(output_indices.begin(), cuda::std::get<0>(it)),
          [offsets = raft::device_span<edge_t>(offsets.data(), offsets.size())] __device__(
            auto pair) { offsets[cuda::std::get<0>(pair)] += cuda::std::get<1>(pair); });
      }
    } else {
      thrust::for_each(handle.get_thrust_policy(),
                       edgelist_major_first,
                       edgelist_major_last,
                       [offsets = raft::device_span<edge_t>(offsets.data(), offsets.size()),
                        major_range_first] __device__(auto major) {
                         cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(
                           offsets[major - major_range_first]);
                         atomic_counter.fetch_add(edge_t{1}, cuda::std::memory_order_relaxed);
                       });
    }

    cugraph::exclusive_scan(
      handle.get_thrust_policy(), offsets.begin(), offsets.end(), offsets.begin());
  }

  return offsets;
}

template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>> compress_hypersparse_offsets(
  raft::handle_t const& handle,
  rmm::device_uvector<edge_t>&& offsets,
  vertex_t major_range_first,
  vertex_t major_hypersparse_first,
  vertex_t major_range_last,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

  auto dcs_nzd_vertices = large_buffer_type
                            ? large_buffer_manager::allocate_memory_buffer<vertex_t>(
                                major_range_last - major_hypersparse_first, handle.get_stream())
                            : rmm::device_uvector<vertex_t>(
                                major_range_last - major_hypersparse_first, handle.get_stream());

  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_counting_iterator(major_hypersparse_first),
                    thrust::make_counting_iterator(major_range_last),
                    dcs_nzd_vertices.begin(),
                    [major_range_first, offsets = offsets.data()] __device__(auto major) {
                      auto major_offset = major - major_range_first;
                      return offsets[major_offset + 1] - offsets[major_offset] > 0 ? major
                                                                                   : invalid_vertex;
                    });

  auto pair_first = thrust::make_zip_iterator(
    dcs_nzd_vertices.begin(), offsets.begin() + (major_hypersparse_first - major_range_first));
  CUGRAPH_EXPECTS(
    dcs_nzd_vertices.size() < static_cast<size_t>(std::numeric_limits<int32_t>::max()),
    "remove_if will fail (https://github.com/NVIDIA/thrust/issues/1302), work-around required.");
  dcs_nzd_vertices.resize(
    cuda::std::distance(pair_first,
                        thrust::remove_if(handle.get_thrust_policy(),
                                          pair_first,
                                          pair_first + dcs_nzd_vertices.size(),
                                          [] __device__(auto pair) {
                                            return cuda::std::get<0>(pair) == invalid_vertex;
                                          })),
    handle.get_stream());
  dcs_nzd_vertices.shrink_to_fit(handle.get_stream());
  if (static_cast<vertex_t>(dcs_nzd_vertices.size()) < major_range_last - major_hypersparse_first) {
    // copying offsets.back() to the new last position
    thrust::copy(
      handle.get_thrust_policy(),
      offsets.begin() + (major_range_last - major_range_first),
      offsets.end(),
      offsets.begin() + (major_hypersparse_first - major_range_first) + dcs_nzd_vertices.size());
    offsets.resize((major_hypersparse_first - major_range_first) + dcs_nzd_vertices.size() + 1,
                   handle.get_stream());
    offsets.shrink_to_fit(handle.get_stream());
  }

  return std::make_tuple(std::move(offsets), std::move(dcs_nzd_vertices));
}

// A packed edge key is (major - major_range_first) << minor_bits | (minor - minor_range_first).
// Its unsigned order is the (major, minor) lexicographic order, so one radix sort replaces the
// comparison merge sort over a two-column zip key. Both radix and merge sort are stable, which
// keeps the input order of parallel edges and their values.

// Returns minor_bits when every edge with endpoints inside the given ranges packs into one
// uint64_t key, std::nullopt otherwise. A minor width of 64 is excluded because the major offset
// would then be shifted by 64.
template <typename vertex_t>
std::optional<int> packed_edge_key_minor_bits(vertex_t major_range_first,
                                              vertex_t major_range_last,
                                              vertex_t minor_range_first,
                                              vertex_t minor_range_last)
{
  auto offset_bits = [](size_t range_size) {
    return range_size > 1 ? static_cast<int>(std::bit_width(range_size - 1)) : 0;
  };
  auto major_bits = offset_bits(static_cast<size_t>(major_range_last - major_range_first));
  auto minor_bits = offset_bits(static_cast<size_t>(minor_range_last - minor_range_first));
  return ((minor_bits < 64) && (major_bits + minor_bits <= 64)) ? std::make_optional(minor_bits)
                                                                : std::nullopt;
}

template <typename vertex_t>
struct pack_edge_key_t {
  vertex_t major_range_first{};
  vertex_t minor_range_first{};
  int minor_bits{};

  __device__ uint64_t operator()(vertex_t major, vertex_t minor) const
  {
    return (static_cast<uint64_t>(major - major_range_first) << minor_bits) |
           static_cast<uint64_t>(minor - minor_range_first);
  }
};

template <typename vertex_t>
struct packed_edge_key_major_t {
  vertex_t major_range_first{};
  int minor_bits{};

  __device__ vertex_t operator()(uint64_t key) const
  {
    return major_range_first + static_cast<vertex_t>(key >> minor_bits);
  }
};

template <typename vertex_t>
struct packed_edge_key_minor_t {
  vertex_t minor_range_first{};
  uint64_t minor_mask{};

  __device__ vertex_t operator()(uint64_t key) const
  {
    return minor_range_first + static_cast<vertex_t>(key & minor_mask);
  }
};

// mem_frugal_partition compares group IDs with an int pivot, which cannot hold every major offset,
// so this op compares the major offset itself and returns group 0 (below the pivot) or 1.
struct packed_edge_key_group_t {
  uint64_t pivot_major_offset{};
  int minor_bits{};

  __device__ int operator()(uint64_t key) const
  {
    return (key >> minor_bits) < pivot_major_offset ? 0 : 1;
  }
};

// 8-byte vertex IDs hold their keys in the minor buffer; 4-byte IDs need a separate key buffer.
template <typename vertex_t>
using packed_edge_key_buffer_t =
  rmm::device_uvector<std::conditional_t<sizeof(vertex_t) == sizeof(uint64_t), vertex_t, uint64_t>>;

template <typename T>
uint64_t* packed_edge_key_data(rmm::device_uvector<T>& keys)
{
  static_assert(sizeof(T) == sizeof(uint64_t));
  return reinterpret_cast<uint64_t*>(keys.data());
}

// Consumes both columns. For 8-byte IDs each key overwrites its own minor element (int64_t and
// uint64_t storage may alias), so the peak is the two input columns. For 4-byte IDs the key buffer
// coexists with both columns until they are released.
template <typename vertex_t>
packed_edge_key_buffer_t<vertex_t> pack_edge_keys(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  vertex_t major_range_first,
  vertex_t minor_range_first,
  int minor_bits,
  std::optional<large_buffer_type_t> large_edge_buffer_type)
{
  pack_edge_key_t<vertex_t> pack{major_range_first, minor_range_first, minor_bits};
  if constexpr (sizeof(vertex_t) == sizeof(uint64_t)) {
    thrust::transform(handle.get_thrust_policy(),
                      majors.begin(),
                      majors.end(),
                      minors.begin(),
                      packed_edge_key_data(minors),
                      pack);
    majors.resize(0, handle.get_stream());
    majors.shrink_to_fit(handle.get_stream());
    return std::move(minors);
  } else {
    auto keys =
      large_edge_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<uint64_t>(majors.size(), handle.get_stream())
        : rmm::device_uvector<uint64_t>(majors.size(), handle.get_stream());
    thrust::transform(
      handle.get_thrust_policy(), majors.begin(), majors.end(), minors.begin(), keys.begin(), pack);
    majors.resize(0, handle.get_stream());
    majors.shrink_to_fit(handle.get_stream());
    minors.resize(0, handle.get_stream());
    minors.shrink_to_fit(handle.get_stream());
    return keys;
  }
}

// Consumes the sorted keys and returns the minors. 8-byte IDs are unpacked in place.
template <typename vertex_t>
rmm::device_uvector<vertex_t> unpack_edge_key_minors(
  raft::handle_t const& handle,
  packed_edge_key_buffer_t<vertex_t>&& keys,
  vertex_t minor_range_first,
  int minor_bits,
  std::optional<large_buffer_type_t> large_edge_buffer_type)
{
  auto key_first = packed_edge_key_data(keys);
  packed_edge_key_minor_t<vertex_t> unpack{minor_range_first, (uint64_t{1} << minor_bits) - 1};
  if constexpr (sizeof(vertex_t) == sizeof(uint64_t)) {
    thrust::transform(
      handle.get_thrust_policy(), key_first, key_first + keys.size(), keys.begin(), unpack);
    return std::move(keys);
  } else {
    auto minors =
      large_edge_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<vertex_t>(keys.size(), handle.get_stream())
        : rmm::device_uvector<vertex_t>(keys.size(), handle.get_stream());
    thrust::transform(
      handle.get_thrust_policy(), key_first, key_first + keys.size(), minors.begin(), unpack);
    keys.resize(0, handle.get_stream());
    keys.shrink_to_fit(handle.get_stream());
    return minors;
  }
}

// compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid
template <typename vertex_t, typename edge_t, typename edge_value_t, bool store_transposed>
std::tuple<rmm::device_uvector<edge_t>,
           rmm::device_uvector<vertex_t>,
           dataframe_buffer_type_t<edge_value_t>,
           std::optional<rmm::device_uvector<vertex_t>>>
sort_and_compress_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  dataframe_buffer_type_t<edge_value_t>&& edgelist_values,
  vertex_t major_range_first,
  std::optional<vertex_t> major_hypersparse_first,
  vertex_t major_range_last,
  vertex_t minor_range_first,
  vertex_t minor_range_last,
  size_t mem_frugal_threshold,
  std::optional<large_buffer_type_t> large_vertex_buffer_type = std::nullopt,
  std::optional<large_buffer_type_t> large_edge_buffer_type   = std::nullopt)
{
  CUGRAPH_EXPECTS((!large_vertex_buffer_type && !large_edge_buffer_type) ||
                    cugraph::large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto edgelist_majors = std::move(store_transposed ? edgelist_dsts : edgelist_srcs);
  auto edgelist_minors = std::move(store_transposed ? edgelist_srcs : edgelist_dsts);

  rmm::device_uvector<edge_t> offsets(0, handle.get_stream());
  rmm::device_uvector<vertex_t> indices(0, handle.get_stream());
  auto values     = allocate_dataframe_buffer<edge_value_t>(0, handle.get_stream());
  auto pair_first = thrust::make_zip_iterator(edgelist_majors.begin(), edgelist_minors.begin());
  auto const minor_bits = packed_edge_key_minor_bits(
    major_range_first, major_range_last, minor_range_first, minor_range_last);
  // With 4-byte IDs, the frugal branch keeps both columns live while it sorts a quarter, and a
  // packed key buffer plus its radix copy add 16 bytes per sorted edge where the merge sort adds 8,
  // so that branch keeps the merge sort.
  if (minor_bits && ((sizeof(vertex_t) == sizeof(uint64_t)) ||
                     (edgelist_minors.size() <= mem_frugal_threshold))) {
    auto keys        = pack_edge_keys(handle,
                               std::move(edgelist_majors),
                               std::move(edgelist_minors),
                               major_range_first,
                               minor_range_first,
                               *minor_bits,
                               large_edge_buffer_type);
    auto key_first   = packed_edge_key_data(keys);
    auto key_last    = key_first + keys.size();
    auto major_first = cuda::make_transform_iterator(
      key_first, packed_edge_key_major_t<vertex_t>{major_range_first, *minor_bits});
    auto value_first = get_dataframe_buffer_begin(edgelist_values);
    if (keys.size() > mem_frugal_threshold) {
      // Same quarter partitioning as the zip-key branch below: values move with their keys, and
      // each sort's workspace covers about a quarter of the edges.
      offsets = compute_sparse_offsets<edge_t>(handle,
                                               major_first,
                                               major_first + keys.size(),
                                               major_range_first,
                                               major_range_last,
                                               false /* edgelist_major_sorted */,
                                               large_vertex_buffer_type);
      std::array<uint64_t, 3> pivots{};
      for (size_t i = 0; i < 3; ++i) {
        pivots[i] = static_cast<uint64_t>(cuda::std::distance(
          offsets.begin(),
          thrust::lower_bound(handle.get_thrust_policy(),
                              offsets.begin(),
                              offsets.end(),
                              static_cast<edge_t>((keys.size() * (i + 1)) / 4))));
      }
      auto second_half_first =
        detail::mem_frugal_partition(key_first,
                                     key_last,
                                     value_first,
                                     packed_edge_key_group_t{pivots[1], *minor_bits},
                                     1,
                                     handle.get_stream(),
                                     large_edge_buffer_type);
      auto second_quarter_first =
        detail::mem_frugal_partition(key_first,
                                     std::get<0>(second_half_first),
                                     value_first,
                                     packed_edge_key_group_t{pivots[0], *minor_bits},
                                     1,
                                     handle.get_stream(),
                                     large_edge_buffer_type);
      auto last_quarter_first =
        detail::mem_frugal_partition(std::get<0>(second_half_first),
                                     key_last,
                                     std::get<1>(second_half_first),
                                     packed_edge_key_group_t{pivots[2], *minor_bits},
                                     1,
                                     handle.get_stream(),
                                     large_edge_buffer_type);
      thrust::sort_by_key(
        handle.get_thrust_policy(), key_first, std::get<0>(second_quarter_first), value_first);
      thrust::sort_by_key(handle.get_thrust_policy(),
                          std::get<0>(second_quarter_first),
                          std::get<0>(second_half_first),
                          std::get<1>(second_quarter_first));
      thrust::sort_by_key(handle.get_thrust_policy(),
                          std::get<0>(second_half_first),
                          std::get<0>(last_quarter_first),
                          std::get<1>(second_half_first));
      thrust::sort_by_key(handle.get_thrust_policy(),
                          std::get<0>(last_quarter_first),
                          key_last,
                          std::get<1>(last_quarter_first));
    } else {
      auto exec_policy =
        large_edge_buffer_type
          ? rmm::exec_policy_nosync(handle.get_stream(), large_buffer_manager::memory_buffer_mr())
          : handle.get_thrust_policy();
      thrust::sort_by_key(exec_policy, key_first, key_last, value_first);

      offsets = compute_sparse_offsets<edge_t>(handle,
                                               major_first,
                                               major_first + keys.size(),
                                               major_range_first,
                                               major_range_last,
                                               true /* edgelist_major_sorted */,
                                               large_vertex_buffer_type);
    }
    edgelist_minors = unpack_edge_key_minors(
      handle, std::move(keys), minor_range_first, *minor_bits, large_edge_buffer_type);
  } else if (edgelist_minors.size() > mem_frugal_threshold) {
    offsets = compute_sparse_offsets<edge_t>(handle,
                                             edgelist_majors.begin(),
                                             edgelist_majors.end(),
                                             major_range_first,
                                             major_range_last,
                                             false /* edgelist_major_sorted */,
                                             large_vertex_buffer_type);

    std::array<vertex_t, 3> pivots{};
    for (size_t i = 0; i < 3; ++i) {
      pivots[i] =
        major_range_first +
        static_cast<vertex_t>(cuda::std::distance(
          offsets.begin(),
          thrust::lower_bound(handle.get_thrust_policy(),
                              offsets.begin(),
                              offsets.end(),
                              static_cast<edge_t>((edgelist_minors.size() * (i + 1)) / 4))));
    }
    auto second_half_first =
      detail::mem_frugal_partition(pair_first,
                                   pair_first + edgelist_minors.size(),
                                   get_dataframe_buffer_begin(edgelist_values),
                                   thrust_tuple_get<cuda::std::tuple<vertex_t, vertex_t>, 0>{},
                                   pivots[1],
                                   handle.get_stream(),
                                   large_edge_buffer_type);
    auto second_quarter_first =
      detail::mem_frugal_partition(pair_first,
                                   std::get<0>(second_half_first),
                                   get_dataframe_buffer_begin(edgelist_values),
                                   thrust_tuple_get<cuda::std::tuple<vertex_t, vertex_t>, 0>{},
                                   pivots[0],
                                   handle.get_stream(),
                                   large_edge_buffer_type);
    auto last_quarter_first =
      detail::mem_frugal_partition(std::get<0>(second_half_first),
                                   pair_first + edgelist_minors.size(),
                                   std::get<1>(second_half_first),
                                   thrust_tuple_get<cuda::std::tuple<vertex_t, vertex_t>, 0>{},
                                   pivots[2],
                                   handle.get_stream(),
                                   large_edge_buffer_type);
    thrust::sort_by_key(handle.get_thrust_policy(),
                        pair_first,
                        std::get<0>(second_quarter_first),
                        get_dataframe_buffer_begin(edgelist_values));
    thrust::sort_by_key(handle.get_thrust_policy(),
                        std::get<0>(second_quarter_first),
                        std::get<0>(second_half_first),
                        std::get<1>(second_quarter_first));
    thrust::sort_by_key(handle.get_thrust_policy(),
                        std::get<0>(second_half_first),
                        std::get<0>(last_quarter_first),
                        std::get<1>(second_half_first));
    thrust::sort_by_key(handle.get_thrust_policy(),
                        std::get<0>(last_quarter_first),
                        pair_first + edgelist_minors.size(),
                        std::get<1>(last_quarter_first));
  } else {
    auto exec_policy =
      large_edge_buffer_type
        ? rmm::exec_policy_nosync(handle.get_stream(), large_buffer_manager::memory_buffer_mr())
        : handle.get_thrust_policy();
    thrust::sort_by_key(exec_policy,
                        pair_first,
                        pair_first + edgelist_minors.size(),
                        get_dataframe_buffer_begin(edgelist_values));

    offsets = compute_sparse_offsets<edge_t>(handle,
                                             edgelist_majors.begin(),
                                             edgelist_majors.end(),
                                             major_range_first,
                                             major_range_last,
                                             true /* edgelist_major_sorted */,
                                             large_vertex_buffer_type);
  }
  indices = std::move(edgelist_minors);
  values  = std::move(edgelist_values);

  edgelist_majors.resize(0, handle.get_stream());
  edgelist_majors.shrink_to_fit(handle.get_stream());

  std::optional<rmm::device_uvector<vertex_t>> dcs_nzd_vertices{std::nullopt};
  if (major_hypersparse_first) {
    std::tie(offsets, dcs_nzd_vertices) = compress_hypersparse_offsets(handle,
                                                                       std::move(offsets),
                                                                       major_range_first,
                                                                       *major_hypersparse_first,
                                                                       major_range_last,
                                                                       large_vertex_buffer_type);
  }

  return std::make_tuple(
    std::move(offsets), std::move(indices), std::move(values), std::move(dcs_nzd_vertices));
}

// compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid
template <typename vertex_t, typename edge_t, bool store_transposed>
std::tuple<rmm::device_uvector<edge_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<vertex_t>>>
sort_and_compress_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  vertex_t major_range_first,
  std::optional<vertex_t> major_hypersparse_first,
  vertex_t major_range_last,
  vertex_t minor_range_first,
  vertex_t minor_range_last,
  size_t mem_frugal_threshold,
  std::optional<large_buffer_type_t> large_vertex_buffer_type = std::nullopt,
  std::optional<large_buffer_type_t> large_edge_buffer_type   = std::nullopt)
{
  auto edgelist_majors = std::move(store_transposed ? edgelist_dsts : edgelist_srcs);
  auto edgelist_minors = std::move(store_transposed ? edgelist_srcs : edgelist_dsts);

  rmm::device_uvector<edge_t> offsets(0, handle.get_stream());
  rmm::device_uvector<vertex_t> indices(0, handle.get_stream());
  auto const minor_bits = packed_edge_key_minor_bits(
    major_range_first, major_range_last, minor_range_first, minor_range_last);
  // 8-byte keys packed in place sort within the input columns' footprint, so quarter partitioning
  // cannot lower the peak. With 4-byte IDs, the frugal branch keeps both columns live while it
  // sorts a quarter, and a packed key buffer plus its radix copy add 16 bytes per sorted edge where
  // the merge sort adds 8, so that branch keeps the merge sort.
  if (minor_bits && ((sizeof(vertex_t) == sizeof(uint64_t)) ||
                     (edgelist_minors.size() <= mem_frugal_threshold))) {
    auto keys      = pack_edge_keys(handle,
                               std::move(edgelist_majors),
                               std::move(edgelist_minors),
                               major_range_first,
                               minor_range_first,
                               *minor_bits,
                               large_edge_buffer_type);
    auto key_first = packed_edge_key_data(keys);
    auto exec_policy =
      large_edge_buffer_type
        ? rmm::exec_policy_nosync(handle.get_stream(), large_buffer_manager::memory_buffer_mr())
        : handle.get_thrust_policy();
    cugraph::sort(exec_policy, key_first, key_first + keys.size());
    auto major_first = cuda::make_transform_iterator(
      key_first, packed_edge_key_major_t<vertex_t>{major_range_first, *minor_bits});
    offsets         = compute_sparse_offsets<edge_t>(handle,
                                             major_first,
                                             major_first + keys.size(),
                                             major_range_first,
                                             major_range_last,
                                             true,
                                             large_vertex_buffer_type);
    edgelist_minors = unpack_edge_key_minors(
      handle, std::move(keys), minor_range_first, *minor_bits, large_edge_buffer_type);
  } else if (edgelist_minors.size() > mem_frugal_threshold) {
    static_assert((sizeof(vertex_t) == 4) || (sizeof(vertex_t) == 8));
    if ((sizeof(vertex_t) == 8) && (static_cast<size_t>(major_range_last - major_range_first) <=
                                    static_cast<size_t>(std::numeric_limits<uint32_t>::max()))) {
      auto edgelist_major_offsets =
        large_edge_buffer_type
          ? large_buffer_manager::allocate_memory_buffer<uint32_t>(edgelist_majors.size(),
                                                                   handle.get_stream())
          : rmm::device_uvector<uint32_t>(edgelist_majors.size(), handle.get_stream());
      thrust::transform(
        handle.get_thrust_policy(),
        edgelist_majors.begin(),
        edgelist_majors.end(),
        edgelist_major_offsets.begin(),
        cuda::proclaim_return_type<uint32_t>([major_range_first] __device__(vertex_t major) {
          return static_cast<uint32_t>(major - major_range_first);
        }));
      edgelist_majors.resize(0, handle.get_stream());
      edgelist_majors.shrink_to_fit(handle.get_stream());

      offsets =
        compute_sparse_offsets<edge_t>(handle,
                                       edgelist_major_offsets.begin(),
                                       edgelist_major_offsets.end(),
                                       uint32_t{0},
                                       static_cast<uint32_t>(major_range_last - major_range_first),
                                       false,
                                       large_vertex_buffer_type);
      std::array<uint32_t, 3> pivots{};
      for (size_t i = 0; i < 3; ++i) {
        pivots[i] = static_cast<uint32_t>(cuda::std::distance(
          offsets.begin(),
          thrust::lower_bound(handle.get_thrust_policy(),
                              offsets.begin(),
                              offsets.end(),
                              static_cast<edge_t>((edgelist_major_offsets.size() * (i + 1)) / 4))));
      }

      auto pair_first =
        thrust::make_zip_iterator(edgelist_major_offsets.begin(), edgelist_minors.begin());
      auto second_half_first =
        detail::mem_frugal_partition(pair_first,
                                     pair_first + edgelist_major_offsets.size(),
                                     thrust_tuple_get<cuda::std::tuple<uint32_t, vertex_t>, 0>{},
                                     pivots[1],
                                     handle.get_stream(),
                                     large_edge_buffer_type);
      auto second_quarter_first =
        detail::mem_frugal_partition(pair_first,
                                     second_half_first,
                                     thrust_tuple_get<cuda::std::tuple<uint32_t, vertex_t>, 0>{},
                                     pivots[0],
                                     handle.get_stream(),
                                     large_edge_buffer_type);
      auto last_quarter_first =
        detail::mem_frugal_partition(second_half_first,
                                     pair_first + edgelist_major_offsets.size(),
                                     thrust_tuple_get<cuda::std::tuple<uint32_t, vertex_t>, 0>{},
                                     pivots[2],
                                     handle.get_stream(),
                                     large_edge_buffer_type);
      cugraph::sort(handle.get_thrust_policy(), pair_first, second_quarter_first);
      cugraph::sort(handle.get_thrust_policy(), second_quarter_first, second_half_first);
      cugraph::sort(handle.get_thrust_policy(), second_half_first, last_quarter_first);
      cugraph::sort(
        handle.get_thrust_policy(), last_quarter_first, pair_first + edgelist_major_offsets.size());
    } else {
      offsets = compute_sparse_offsets<edge_t>(handle,
                                               edgelist_majors.begin(),
                                               edgelist_majors.end(),
                                               major_range_first,
                                               major_range_last,
                                               false,
                                               large_vertex_buffer_type);
      std::array<vertex_t, 3> pivots{};
      for (size_t i = 0; i < 3; ++i) {
        pivots[i] =
          major_range_first +
          static_cast<vertex_t>(cuda::std::distance(
            offsets.begin(),
            thrust::lower_bound(handle.get_thrust_policy(),
                                offsets.begin(),
                                offsets.end(),
                                static_cast<edge_t>((edgelist_minors.size() * (i + 1)) / 4))));
      }
      auto edge_first = thrust::make_zip_iterator(edgelist_majors.begin(), edgelist_minors.begin());
      auto second_half_first =
        detail::mem_frugal_partition(edge_first,
                                     edge_first + edgelist_majors.size(),
                                     thrust_tuple_get<cuda::std::tuple<vertex_t, vertex_t>, 0>{},
                                     pivots[1],
                                     handle.get_stream(),
                                     large_edge_buffer_type);
      auto second_quarter_first =
        detail::mem_frugal_partition(edge_first,
                                     second_half_first,
                                     thrust_tuple_get<cuda::std::tuple<vertex_t, vertex_t>, 0>{},
                                     pivots[0],
                                     handle.get_stream(),
                                     large_edge_buffer_type);
      auto last_quarter_first =
        detail::mem_frugal_partition(second_half_first,
                                     edge_first + edgelist_majors.size(),
                                     thrust_tuple_get<cuda::std::tuple<vertex_t, vertex_t>, 0>{},
                                     pivots[2],
                                     handle.get_stream(),
                                     large_edge_buffer_type);
      cugraph::sort(handle.get_thrust_policy(), edge_first, second_quarter_first);
      cugraph::sort(handle.get_thrust_policy(), second_quarter_first, second_half_first);
      cugraph::sort(handle.get_thrust_policy(), second_half_first, last_quarter_first);
      cugraph::sort(
        handle.get_thrust_policy(), last_quarter_first, edge_first + edgelist_majors.size());
      edgelist_majors.resize(0, handle.get_stream());
      edgelist_majors.shrink_to_fit(handle.get_stream());
    }
  } else {
    auto edge_first = thrust::make_zip_iterator(edgelist_majors.begin(), edgelist_minors.begin());
    auto exec_policy =
      large_edge_buffer_type
        ? rmm::exec_policy_nosync(handle.get_stream(), large_buffer_manager::memory_buffer_mr())
        : handle.get_thrust_policy();
    cugraph::sort(exec_policy, edge_first, edge_first + edgelist_minors.size());
    offsets = compute_sparse_offsets<edge_t>(handle,
                                             edgelist_majors.begin(),
                                             edgelist_majors.end(),
                                             major_range_first,
                                             major_range_last,
                                             true,
                                             large_vertex_buffer_type);
    edgelist_majors.resize(0, handle.get_stream());
    edgelist_majors.shrink_to_fit(handle.get_stream());
  }
  indices = std::move(edgelist_minors);

  std::optional<rmm::device_uvector<vertex_t>> dcs_nzd_vertices{std::nullopt};
  if (major_hypersparse_first) {
    std::tie(offsets, dcs_nzd_vertices) = compress_hypersparse_offsets(handle,
                                                                       std::move(offsets),
                                                                       major_range_first,
                                                                       *major_hypersparse_first,
                                                                       major_range_last,
                                                                       large_vertex_buffer_type);
  }

  return std::make_tuple(std::move(offsets), std::move(indices), std::move(dcs_nzd_vertices));
}

template <typename edge_t, typename VertexIterator, typename EdgeValueIterator>
void sort_adjacency_list(raft::handle_t const& handle,
                         raft::device_span<edge_t const> offsets,
                         VertexIterator index_first /* [INOUT] */,
                         VertexIterator index_last /* [INOUT] */,
                         EdgeValueIterator edge_value_first /* [INOUT] */)
{
  using vertex_t     = typename thrust::iterator_traits<VertexIterator>::value_type;
  using edge_value_t = typename thrust::iterator_traits<EdgeValueIterator>::value_type;

  // 1. Check if there is anything to sort

  auto num_edges = static_cast<edge_t>(cuda::std::distance(index_first, index_last));
  if (num_edges == 0) { return; }

  // 2. We segmented sort edges in chunks, and we need to adjust chunk offsets as we need to sort
  // each vertex's neighbors at once.

  // to limit memory footprint ((1 << 20) is a tuning parameter)
  auto approx_edges_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 20);
  auto [h_vertex_offsets, h_edge_offsets] = detail::compute_offset_aligned_element_chunks(
    handle, offsets, num_edges, approx_edges_to_sort_per_iteration);
  auto num_chunks = h_vertex_offsets.size() - 1;

  // 3. Segmented sort each vertex's neighbors

  size_t max_chunk_size{0};
  for (size_t i = 0; i < num_chunks; ++i) {
    max_chunk_size =
      std::max(max_chunk_size, static_cast<size_t>(h_edge_offsets[i + 1] - h_edge_offsets[i]));
  }
  rmm::device_uvector<vertex_t> segment_sorted_indices(max_chunk_size, handle.get_stream());
  rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
  auto segment_sorted_values =
    allocate_dataframe_buffer<edge_value_t>(max_chunk_size, handle.get_stream());
  if constexpr (std::is_arithmetic_v<edge_value_t>) {
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t tmp_storage_bytes{0};
      auto offset_first = cuda::make_transform_iterator(offsets.data() + h_vertex_offsets[i],
                                                        shift_left_t<edge_t>{h_edge_offsets[i]});
      cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                          tmp_storage_bytes,
                                          index_first + h_edge_offsets[i],
                                          segment_sorted_indices.data(),
                                          edge_value_first + h_edge_offsets[i],
                                          segment_sorted_values.data(),
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }
      cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                          tmp_storage_bytes,
                                          index_first + h_edge_offsets[i],
                                          segment_sorted_indices.data(),
                                          edge_value_first + h_edge_offsets[i],
                                          segment_sorted_values.data(),
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   segment_sorted_indices.begin(),
                   segment_sorted_indices.begin() + (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                   index_first + h_edge_offsets[i]);
      thrust::copy(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(segment_sorted_values),
                   get_dataframe_buffer_begin(segment_sorted_values) +
                     (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                   edge_value_first + h_edge_offsets[i]);
    }
  } else {  // cub's segmented sort does not support thrust iterators (so we can't directly sort
            // edge values with thrust::zip_iterator)
    rmm::device_uvector<edge_t> input_edge_value_offsets(max_chunk_size, handle.get_stream());
    rmm::device_uvector<edge_t> segment_sorted_edge_value_offsets(max_chunk_size,
                                                                  handle.get_stream());
    cugraph::sequence(handle.get_thrust_policy(),
                      input_edge_value_offsets.begin(),
                      input_edge_value_offsets.end(),
                      edge_t{0});
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t tmp_storage_bytes{0};
      auto offset_first = cuda::make_transform_iterator(offsets.data() + h_vertex_offsets[i],
                                                        shift_left_t<edge_t>{h_edge_offsets[i]});
      cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                          tmp_storage_bytes,
                                          index_first + h_edge_offsets[i],
                                          segment_sorted_indices.data(),
                                          input_edge_value_offsets.data(),
                                          segment_sorted_edge_value_offsets.data(),
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }
      cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                          tmp_storage_bytes,
                                          index_first + h_edge_offsets[i],
                                          segment_sorted_indices.data(),
                                          input_edge_value_offsets.data(),
                                          segment_sorted_edge_value_offsets.data(),
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   segment_sorted_indices.begin(),
                   segment_sorted_indices.begin() + (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                   index_first + h_edge_offsets[i]);
      cugraph::gather(
        handle.get_thrust_policy(),
        segment_sorted_edge_value_offsets.begin(),
        segment_sorted_edge_value_offsets.begin() + (h_edge_offsets[i + 1] - h_edge_offsets[i]),
        edge_value_first + h_edge_offsets[i],
        get_dataframe_buffer_begin(segment_sorted_values));
      thrust::copy(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(segment_sorted_values),
                   get_dataframe_buffer_begin(segment_sorted_values) +
                     (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                   edge_value_first + h_edge_offsets[i]);
    }
  }
}

template <typename edge_t, typename VertexIterator>
void sort_adjacency_list(raft::handle_t const& handle,
                         raft::device_span<edge_t const> offsets,
                         VertexIterator index_first /* [INOUT] */,
                         VertexIterator index_last /* [INOUT] */)
{
  using vertex_t = typename thrust::iterator_traits<VertexIterator>::value_type;

  // 1. Check if there is anything to sort

  auto num_edges = static_cast<edge_t>(cuda::std::distance(index_first, index_last));
  if (num_edges == 0) { return; }

  // 2. We segmented sort edges in chunks, and we need to adjust chunk offsets as we need to sort
  // each vertex's neighbors at once.

  // to limit memory footprint ((1 << 20) is a tuning parameter)
  auto approx_edges_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 20);
  auto [h_vertex_offsets, h_edge_offsets] = detail::compute_offset_aligned_element_chunks(
    handle, offsets, num_edges, approx_edges_to_sort_per_iteration);
  auto num_chunks = h_vertex_offsets.size() - 1;

  // 3. Segmented sort each vertex's neighbors

  size_t max_chunk_size{0};
  for (size_t i = 0; i < num_chunks; ++i) {
    max_chunk_size =
      std::max(max_chunk_size, static_cast<size_t>(h_edge_offsets[i + 1] - h_edge_offsets[i]));
  }
  rmm::device_uvector<vertex_t> segment_sorted_indices(max_chunk_size, handle.get_stream());
  rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
  for (size_t i = 0; i < num_chunks; ++i) {
    size_t tmp_storage_bytes{0};
    auto offset_first = cuda::make_transform_iterator(offsets.data() + h_vertex_offsets[i],
                                                      shift_left_t<edge_t>{h_edge_offsets[i]});
    cub::DeviceSegmentedSort::SortKeys(static_cast<void*>(nullptr),
                                       tmp_storage_bytes,
                                       index_first + h_edge_offsets[i],
                                       segment_sorted_indices.data(),
                                       h_edge_offsets[i + 1] - h_edge_offsets[i],
                                       h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                       offset_first,
                                       offset_first + 1,
                                       handle.get_stream());
    if (tmp_storage_bytes > d_tmp_storage.size()) {
      d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
    }
    cub::DeviceSegmentedSort::SortKeys(d_tmp_storage.data(),
                                       tmp_storage_bytes,
                                       index_first + h_edge_offsets[i],
                                       segment_sorted_indices.data(),
                                       h_edge_offsets[i + 1] - h_edge_offsets[i],
                                       h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                       offset_first,
                                       offset_first + 1,
                                       handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 segment_sorted_indices.begin(),
                 segment_sorted_indices.begin() + (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                 index_first + h_edge_offsets[i]);
  }
}


}  // namespace detail
}  // namespace cugraph
