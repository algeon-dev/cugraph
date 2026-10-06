/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "c_api/abstract_functor.hpp"
#include "c_api/graph.hpp"
#include "c_api/graph_helper.hpp"
#include "c_api/paths_result.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/result_factory.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace cugraph {
namespace c_api {

template <typename vertex_t, bool multi_gpu>
size_t count_invalid_vertices(raft::handle_t const& handle,
                              rmm::device_uvector<vertex_t> const& vertices)
{
  auto invalid_count = cugraph::detail::count_values(
    handle,
    raft::device_span<vertex_t const>{vertices.data(), vertices.size()},
    cugraph::invalid_vertex_id<vertex_t>::value);

  if constexpr (multi_gpu) {
    invalid_count = cugraph::host_scalar_allreduce(
      handle.get_comms(), invalid_count, raft::comms::op_t::SUM, handle.get_stream());
  }

  return invalid_count;
}

template <typename vertex_t, bool multi_gpu, typename GraphViewType>
rmm::device_uvector<vertex_t> copy_and_renumber_vertices(
  raft::handle_t const& handle,
  cugraph_type_erased_device_array_view_t const* input,
  GraphViewType const& graph_view,
  rmm::device_uvector<vertex_t> const& number_map,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> vertices(input->size_, handle.get_stream());
  cugraph::c_api::copy_or_transform<vertex_t>(
    raft::device_span<vertex_t>{vertices.data(), vertices.size()}, input, handle.get_stream());

  if constexpr (multi_gpu) {
    std::tie(vertices, std::ignore) = shuffle_ext_vertices(
      handle, std::move(vertices), std::vector<cugraph::arithmetic_device_uvector_t>{});
  }

  renumber_ext_vertices<vertex_t, multi_gpu>(handle,
                                             vertices.data(),
                                             vertices.size(),
                                             number_map.data(),
                                             graph_view.local_vertex_partition_range_first(),
                                             graph_view.local_vertex_partition_range_last(),
                                             do_expensive_check);

  return vertices;
}

template <typename vertex_t, typename GraphViewType>
rmm::device_uvector<uint32_t> make_vertex_bitmap(raft::handle_t const& handle,
                                                 GraphViewType const& graph_view,
                                                 rmm::device_uvector<vertex_t> const& vertices,
                                                 bool default_value,
                                                 bool set_value)
{
  auto const num_local_vertices =
    static_cast<size_t>(graph_view.local_vertex_partition_range_size());
  rmm::device_uvector<uint32_t> bitmap(cugraph::packed_bool_size(num_local_vertices),
                                       handle.get_stream());
  std::vector<uint32_t> h_bitmap(bitmap.size(),
                                 default_value ? cugraph::packed_bool_full_mask()
                                               : cugraph::packed_bool_empty_mask());
  std::vector<vertex_t> h_vertices(vertices.size());
  raft::update_host(h_vertices.data(), vertices.data(), vertices.size(), handle.get_stream());
  handle.sync_stream();

  auto const vertex_first = graph_view.local_vertex_partition_range_first();
  for (auto v : h_vertices) {
    auto const v_offset = v - vertex_first;
    if (set_value) {
      h_bitmap[cugraph::packed_bool_offset(v_offset)] |= cugraph::packed_bool_mask(v_offset);
    } else {
      h_bitmap[cugraph::packed_bool_offset(v_offset)] &= ~cugraph::packed_bool_mask(v_offset);
    }
  }

  raft::update_device(bitmap.data(), h_bitmap.data(), h_bitmap.size(), handle.get_stream());

  return bitmap;
}

template <typename vertex_t>
size_t count_rejected_sources(raft::handle_t const& handle,
                              rmm::device_uvector<vertex_t> const& sources,
                              raft::device_span<uint32_t const> vertex_allow_bitmap,
                              vertex_t vertex_first)
{
  std::vector<vertex_t> h_sources(sources.size());
  std::vector<uint32_t> h_bitmap(vertex_allow_bitmap.size());
  raft::update_host(h_sources.data(), sources.data(), sources.size(), handle.get_stream());
  raft::update_host(h_bitmap.data(),
                    vertex_allow_bitmap.data(),
                    vertex_allow_bitmap.size(),
                    handle.get_stream());
  handle.sync_stream();

  return std::count_if(h_sources.begin(), h_sources.end(), [vertex_first, &h_bitmap](auto v) {
    auto const v_offset = v - vertex_first;
    auto const word     = h_bitmap[cugraph::packed_bool_offset(v_offset)];
    return (word & cugraph::packed_bool_mask(v_offset)) == cugraph::packed_bool_empty_mask();
  });
}

template <typename edge_t, typename GraphViewType>
edge_property_t<edge_t, bool> make_edge_id_include_mask(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  cugraph_type_erased_device_array_view_t const* include_edge_ids,
  edge_property_t<edge_t, edge_t> const& graph_edge_ids);

extern template edge_property_t<int32_t, bool>
make_edge_id_include_mask<int32_t, graph_view_t<int32_t, int32_t, false, false>>(
  raft::handle_t const&,
  graph_view_t<int32_t, int32_t, false, false> const&,
  cugraph_type_erased_device_array_view_t const*,
  edge_property_t<int32_t, int32_t> const&);

extern template edge_property_t<int64_t, bool>
make_edge_id_include_mask<int64_t, graph_view_t<int64_t, int64_t, false, false>>(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, false> const&,
  cugraph_type_erased_device_array_view_t const*,
  edge_property_t<int64_t, int64_t> const&);

struct bfs_functor : public abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_t* graph_;
  cugraph_type_erased_device_array_view_t* sources_;
  bool direction_optimizing_;
  size_t depth_limit_;
  bool compute_predecessors_;
  bool do_expensive_check_;
  cugraph_paths_result_t* result_{};

  bfs_functor(::cugraph_resource_handle_t const* handle,
              ::cugraph_graph_t* graph,
              ::cugraph_type_erased_device_array_view_t* sources,
              bool direction_optimizing,
              size_t depth_limit,
              bool compute_predecessors,
              bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      sources_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(sources)),
      direction_optimizing_(direction_optimizing),
      depth_limit_(depth_limit),
      compute_predecessors_(compute_predecessors),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            typename time_stamp_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // BFS expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> distances(graph_view.local_vertex_partition_range_size(),
                                              handle_.get_stream());
      rmm::device_uvector<vertex_t> predecessors(0, handle_.get_stream());

      if (compute_predecessors_) {
        predecessors.resize(graph_view.local_vertex_partition_range_size(), handle_.get_stream());
      }

      rmm::device_uvector<vertex_t> sources(sources_->size_, handle_.get_stream());
      raft::copy(
        sources.data(), sources_->as_type<vertex_t>(), sources_->size_, handle_.get_stream());

      if constexpr (multi_gpu) {
        std::tie(sources, std::ignore) = shuffle_ext_vertices(
          handle_, std::move(sources), std::vector<cugraph::arithmetic_device_uvector_t>{});
      }

      //
      // Need to renumber sources
      //
      renumber_ext_vertices<vertex_t, multi_gpu>(handle_,
                                                 sources.data(),
                                                 sources.size(),
                                                 number_map->data(),
                                                 graph_view.local_vertex_partition_range_first(),
                                                 graph_view.local_vertex_partition_range_last(),
                                                 do_expensive_check_);

      size_t invalid_count = cugraph::detail::count_values(
        handle_,
        raft::device_span<vertex_t const>{sources.data(), sources.size()},
        cugraph::invalid_vertex_id<vertex_t>::value);

      if constexpr (multi_gpu) {
        invalid_count = cugraph::host_scalar_allreduce(
          handle_.get_comms(), invalid_count, raft::comms::op_t::SUM, handle_.get_stream());
      }

      if (invalid_count != 0) {
        mark_error(CUGRAPH_INVALID_INPUT, "BFS sources contain a vertex that is not in the graph");
        return;
      }

      cugraph::bfs<vertex_t, edge_t, multi_gpu>(
        handle_,
        graph_view,
        distances.data(),
        compute_predecessors_ ? predecessors.data() : nullptr,
        sources.data(),
        sources.size(),
        direction_optimizing_,
        static_cast<vertex_t>(depth_limit_),
        do_expensive_check_);

      rmm::device_uvector<vertex_t> vertex_ids(graph_view.local_vertex_partition_range_size(),
                                               handle_.get_stream());
      raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

      if (compute_predecessors_) {
        unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                     predecessors.data(),
                                                     predecessors.size(),
                                                     number_map->data(),
                                                     graph_view.vertex_partition_range_lasts(),
                                                     do_expensive_check_);
      }

      result_ = cugraph::c_api::result_factory::make_paths_result(
                  vertex_ids, distances, predecessors, graph_->vertex_type_)
                  .release();
    }
  }
};

struct bfs_predicate_functor : public abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_t* graph_;
  cugraph_type_erased_device_array_view_t* sources_;
  cugraph_type_erased_device_array_view_t const* include_vertices_;
  cugraph_type_erased_device_array_view_t const* exclude_vertices_;
  cugraph_type_erased_device_array_view_t const* target_vertices_;
  cugraph_type_erased_device_array_view_t const* include_edge_ids_;
  bool direction_optimizing_;
  size_t depth_limit_;
  bool compute_predecessors_;
  bool stop_on_first_target_;
  bool do_expensive_check_;
  bool return_predicate_result_;
  cugraph_paths_result_t* result_{};
  cugraph_bfs_predicate_result_t* predicate_result_{};

  bfs_predicate_functor(
    ::cugraph_resource_handle_t const* handle,
    ::cugraph_graph_t* graph,
    ::cugraph_type_erased_device_array_view_t* sources,
    ::cugraph_type_erased_device_array_view_t const* include_vertices,
    ::cugraph_type_erased_device_array_view_t const* exclude_vertices,
    ::cugraph_type_erased_device_array_view_t const* target_vertices,
    ::cugraph_type_erased_device_array_view_t const* include_edge_ids,
    bool direction_optimizing,
    size_t depth_limit,
    bool compute_predecessors,
    bool stop_on_first_target,
    bool do_expensive_check,
    bool return_predicate_result)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      sources_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(sources)),
      include_vertices_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        include_vertices)),
      exclude_vertices_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        exclude_vertices)),
      target_vertices_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        target_vertices)),
      include_edge_ids_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        include_edge_ids)),
      direction_optimizing_(direction_optimizing),
      depth_limit_(depth_limit),
      compute_predecessors_(compute_predecessors),
      stop_on_first_target_(stop_on_first_target),
      do_expensive_check_(do_expensive_check),
      return_predicate_result_(return_predicate_result)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            typename time_stamp_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    if constexpr (multi_gpu) {
      mark_error(CUGRAPH_NOT_IMPLEMENTED,
                 "cugraph_bfs_with_predicates currently supports single-GPU graphs only");
    } else if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);
      auto graph_view = graph->view();
      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> distances(graph_view.local_vertex_partition_range_size(),
                                              handle_.get_stream());
      rmm::device_uvector<vertex_t> predecessors(0, handle_.get_stream());

      if (compute_predecessors_) {
        predecessors.resize(graph_view.local_vertex_partition_range_size(), handle_.get_stream());
      }

      auto sources = copy_and_renumber_vertices<vertex_t, multi_gpu>(
        handle_, sources_, graph_view, *number_map, do_expensive_check_);

      if (count_invalid_vertices<vertex_t, multi_gpu>(handle_, sources) != 0) {
        mark_error(CUGRAPH_INVALID_INPUT, "BFS sources contain a vertex that is not in the graph");
        return;
      }

      std::optional<rmm::device_uvector<uint32_t>> vertex_allow_bitmap{std::nullopt};
      if (include_vertices_ != nullptr) {
        auto include_vertices = copy_and_renumber_vertices<vertex_t, multi_gpu>(
          handle_, include_vertices_, graph_view, *number_map, do_expensive_check_);
        if (count_invalid_vertices<vertex_t, multi_gpu>(handle_, include_vertices) != 0) {
          mark_error(CUGRAPH_INVALID_INPUT, "Found invalid vertex in include_vertices");
          return;
        }
        vertex_allow_bitmap = make_vertex_bitmap(handle_, graph_view, include_vertices, false, true);
      } else if (exclude_vertices_ != nullptr) {
        auto exclude_vertices = copy_and_renumber_vertices<vertex_t, multi_gpu>(
          handle_, exclude_vertices_, graph_view, *number_map, do_expensive_check_);
        if (count_invalid_vertices<vertex_t, multi_gpu>(handle_, exclude_vertices) != 0) {
          mark_error(CUGRAPH_INVALID_INPUT, "Found invalid vertex in exclude_vertices");
          return;
        }
        vertex_allow_bitmap = make_vertex_bitmap(handle_, graph_view, exclude_vertices, true, false);
      }

      if (vertex_allow_bitmap) {
        auto rejected_sources = count_rejected_sources(
          handle_,
          sources,
          raft::device_span<uint32_t const>{vertex_allow_bitmap->data(), vertex_allow_bitmap->size()},
          graph_view.local_vertex_partition_range_first());
        if (rejected_sources != 0) {
          mark_error(CUGRAPH_INVALID_INPUT, "sources must satisfy the vertex predicate");
          return;
        }
      }

      std::optional<rmm::device_uvector<uint32_t>> target_bitmap{std::nullopt};
      if (target_vertices_ != nullptr) {
        auto target_vertices = copy_and_renumber_vertices<vertex_t, multi_gpu>(
          handle_, target_vertices_, graph_view, *number_map, do_expensive_check_);
        if (count_invalid_vertices<vertex_t, multi_gpu>(handle_, target_vertices) != 0) {
          mark_error(CUGRAPH_INVALID_INPUT, "Found invalid vertex in target_vertices");
          return;
        }
        target_bitmap = make_vertex_bitmap(handle_, graph_view, target_vertices, false, true);
      }

      std::optional<edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
      if (include_edge_ids_ != nullptr) {
        if (graph_->edge_ids_ == nullptr) {
          mark_error(CUGRAPH_INVALID_INPUT, "include_edge_ids requires a graph with edge IDs");
          return;
        }
        auto graph_edge_ids =
          reinterpret_cast<cugraph::edge_property_t<edge_t, edge_t>*>(graph_->edge_ids_);
        edge_mask = make_edge_id_include_mask(handle_, graph_view, include_edge_ids_, *graph_edge_ids);
      }

      cugraph::bfs_predicate_result_t<vertex_t> predicate_metadata{};
      cugraph::bfs_with_predicates<vertex_t, edge_t>(
        handle_,
        graph_view,
        distances.data(),
        compute_predecessors_ ? predecessors.data() : nullptr,
        sources.data(),
        sources.size(),
        edge_mask ? std::make_optional(edge_mask->view()) : std::nullopt,
        vertex_allow_bitmap
          ? std::make_optional(
              raft::device_span<uint32_t const>{vertex_allow_bitmap->data(), vertex_allow_bitmap->size()})
          : std::nullopt,
        target_bitmap
          ? std::make_optional(raft::device_span<uint32_t const>{target_bitmap->data(),
                                                                 target_bitmap->size()})
          : std::nullopt,
        stop_on_first_target_,
        direction_optimizing_,
        static_cast<vertex_t>(depth_limit_),
        do_expensive_check_,
        return_predicate_result_ ? &predicate_metadata : nullptr);

      rmm::device_uvector<vertex_t> vertex_ids(graph_view.local_vertex_partition_range_size(),
                                               handle_.get_stream());
      raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

      if (compute_predecessors_) {
        unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                     predecessors.data(),
                                                     predecessors.size(),
                                                     number_map->data(),
                                                     graph_view.vertex_partition_range_lasts(),
                                                     do_expensive_check_);
      }

      std::unique_ptr<cugraph_bfs_predicate_result_t> predicate_result;
      if (return_predicate_result_) {
        predicate_result = cugraph::c_api::result_factory::make_bfs_predicate_result(
          predicate_metadata.target_found,
          static_cast<size_t>(predicate_metadata.target_distance));
      }

      auto result = cugraph::c_api::result_factory::make_paths_result(
        vertex_ids, distances, predecessors, graph_->vertex_type_);

      predicate_result_ = predicate_result.release();
      result_ = result.release();
    }
  }
};

}  // namespace c_api
}  // namespace cugraph

extern "C" cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_vertices(
  cugraph_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertex_ids_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_distances(
  cugraph_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->distances_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_predecessors(
  cugraph_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->predecessors_->view());
}

extern "C" void cugraph_paths_result_free(cugraph_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t*>(result);
  delete internal_pointer->vertex_ids_;
  delete internal_pointer->distances_;
  delete internal_pointer->predecessors_;
  delete internal_pointer;
}

extern "C" bool_t cugraph_bfs_predicate_result_get_target_found(
  cugraph_bfs_predicate_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_bfs_predicate_result_t*>(result);
  return internal_pointer->target_found_ ? TRUE : FALSE;
}

extern "C" size_t cugraph_bfs_predicate_result_get_target_distance(
  cugraph_bfs_predicate_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_bfs_predicate_result_t*>(result);
  return internal_pointer->target_distance_;
}

extern "C" void cugraph_bfs_predicate_result_free(cugraph_bfs_predicate_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_bfs_predicate_result_t*>(result);
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_bfs(const cugraph_resource_handle_t* handle,
                                            cugraph_graph_t* graph,
                                            cugraph_type_erased_device_array_view_t* sources,
                                            bool_t direction_optimizing,
                                            size_t depth_limit,
                                            bool_t compute_predecessors,
                                            bool_t do_expensive_check,
                                            cugraph_paths_result_t** result,
                                            cugraph_error_t** error)
{
  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(sources)
        ->type_,
    CUGRAPH_INVALID_INPUT,
    "vertex type of graph and sources must match",
    *error);

  cugraph::c_api::bfs_functor functor(handle,
                                      graph,
                                      sources,
                                      direction_optimizing,
                                      depth_limit,
                                      compute_predecessors,
                                      do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_bfs_with_predicates(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  cugraph_type_erased_device_array_view_t* sources,
  const cugraph_type_erased_device_array_view_t* include_vertices,
  const cugraph_type_erased_device_array_view_t* exclude_vertices,
  const cugraph_type_erased_device_array_view_t* target_vertices,
  const cugraph_type_erased_device_array_view_t* include_edge_ids,
  bool_t direction_optimizing,
  size_t depth_limit,
  bool_t compute_predecessors,
  bool_t stop_on_first_target,
  bool_t do_expensive_check,
  cugraph_paths_result_t** result,
  cugraph_bfs_predicate_result_t** predicate_result,
  cugraph_error_t** error)
{
  if (predicate_result != nullptr) { *predicate_result = nullptr; }

  auto p_graph =
    reinterpret_cast<cugraph::c_api::cugraph_graph_t const*>(graph);
  auto p_sources =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(sources);
  auto p_include_vertices =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
      include_vertices);
  auto p_exclude_vertices =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
      exclude_vertices);
  auto p_target_vertices =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
      target_vertices);
  auto p_include_edge_ids =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
      include_edge_ids);

  CAPI_EXPECTS(p_graph->vertex_type_ == p_sources->type_,
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and sources must match",
               *error);
  CAPI_EXPECTS((include_vertices == nullptr) || (p_graph->vertex_type_ == p_include_vertices->type_),
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and include_vertices must match",
               *error);
  CAPI_EXPECTS((exclude_vertices == nullptr) || (p_graph->vertex_type_ == p_exclude_vertices->type_),
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and exclude_vertices must match",
               *error);
  CAPI_EXPECTS((target_vertices == nullptr) || (p_graph->vertex_type_ == p_target_vertices->type_),
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and target_vertices must match",
               *error);
  // Edge IDs are stored as edge_property_t<edge_t, edge_t>, i.e. the edge-ID value type
  // is the same as the graph's edge_t. We therefore compare against edge_type_ here.
  // If edge IDs ever get a dedicated value-type field on cugraph_graph_t, switch to that.
  CAPI_EXPECTS((include_edge_ids == nullptr) || (p_graph->edge_type_ == p_include_edge_ids->type_),
               CUGRAPH_INVALID_INPUT,
               "edge type of graph and include_edge_ids must match",
               *error);
  CAPI_EXPECTS((include_vertices == nullptr) || (exclude_vertices == nullptr),
               CUGRAPH_INVALID_INPUT,
               "include_vertices and exclude_vertices are mutually exclusive",
               *error);

  cugraph::c_api::bfs_predicate_functor functor(handle,
                                                graph,
                                                sources,
                                                include_vertices,
                                                exclude_vertices,
                                                target_vertices,
                                                include_edge_ids,
                                                direction_optimizing,
                                                depth_limit,
                                                compute_predecessors,
                                                stop_on_first_target,
                                                do_expensive_check,
                                                predicate_result != nullptr);

  auto ret_code = cugraph::c_api::run_algorithm(graph, functor, result, error);
  if ((ret_code == CUGRAPH_SUCCESS) && (predicate_result != nullptr)) {
    *predicate_result =
      reinterpret_cast<cugraph_bfs_predicate_result_t*>(functor.predicate_result_);
  }
  return ret_code;
}
