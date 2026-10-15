/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/abstract_functor.hpp"
#include "c_api/error.hpp"
#include "c_api/graph.hpp"
#include "c_api/paths_result.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <limits>

namespace cugraph {
namespace c_api {

struct cugraph_extract_paths_result_t {
  size_t max_path_length_;
  cugraph_type_erased_device_array_t* paths_;
};

struct extract_paths_functor : public abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_t* graph_;
  cugraph_type_erased_device_array_view_t const* sources_;
  cugraph_paths_result_t const* paths_result_;
  cugraph_type_erased_device_array_view_t const* destinations_;
  cugraph_extract_paths_result_t* result_{};

  extract_paths_functor(::cugraph_resource_handle_t const* handle,
                        ::cugraph_graph_t* graph,
                        ::cugraph_type_erased_device_array_view_t const* sources,
                        ::cugraph_paths_result_t const* paths_result,
                        ::cugraph_type_erased_device_array_view_t const* destinations)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      sources_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(sources)),
      paths_result_(reinterpret_cast<cugraph::c_api::cugraph_paths_result_t const*>(paths_result)),
      destinations_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          destinations))
  {
  }

  template <typename vertex_t, typename edge_t>
  void extract_weighted_paths(
    cugraph::graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
    rmm::device_uvector<vertex_t> const& number_map)
  {
    auto const vertex_count = graph_view.local_vertex_partition_range_size();
    if (sources_->size_ != 1) {
      mark_error(CUGRAPH_INVALID_INPUT, "SSSP path extraction requires exactly one source vertex");
      return;
    }
    if (sources_->type_ != graph_->vertex_type_ ||
        destinations_->type_ != graph_->vertex_type_ ||
        paths_result_->vertex_ids_->type_ != graph_->vertex_type_ ||
        paths_result_->predecessors_->type_ != graph_->vertex_type_) {
      mark_error(CUGRAPH_INVALID_INPUT, "SSSP path extraction vertex dtypes must match the graph");
      return;
    }
    if (paths_result_->vertex_ids_->size_ != paths_result_->predecessors_->size_) {
      mark_error(CUGRAPH_INVALID_STATE,
                 "SSSP result returned mismatched vertex and predecessor lengths");
      return;
    }
    auto const invalid = cugraph::invalid_vertex_id<vertex_t>::value;
    auto copy_and_renumber = [&](vertex_t const* data, size_t size) {
      rmm::device_uvector<vertex_t> values(size, handle_.get_stream());
      raft::copy(values.data(), data, size, handle_.get_stream());
      cugraph::renumber_ext_vertices<vertex_t, false>(
        handle_, values.data(), values.size(), number_map.data(),
        graph_view.local_vertex_partition_range_first(),
        graph_view.local_vertex_partition_range_last(), false);
      return values;
    };
    auto sources = copy_and_renumber(sources_->as_type<vertex_t>(), sources_->size_);
    auto destinations = copy_and_renumber(destinations_->as_type<vertex_t>(), destinations_->size_);
    auto vertices = copy_and_renumber(paths_result_->vertex_ids_->as_type<vertex_t>(),
                                      paths_result_->vertex_ids_->size_);
    if (thrust::count(handle_.get_thrust_policy(), sources.begin(), sources.end(), invalid) != 0) {
      mark_error(CUGRAPH_INVALID_INPUT, "SSSP path source vertex is not in the graph");
      return;
    }
    if (thrust::count(handle_.get_thrust_policy(), vertices.begin(), vertices.end(), invalid) != 0) {
      mark_error(CUGRAPH_INVALID_STATE, "SSSP predecessor result contains a vertex outside the graph");
      return;
    }
    auto predecessors = copy_and_renumber(paths_result_->predecessors_->as_type<vertex_t>(),
                                          paths_result_->predecessors_->size_);
    thrust::transform(
      handle_.get_thrust_policy(), paths_result_->predecessors_->as_type<vertex_t>(),
      paths_result_->predecessors_->as_type<vertex_t>() + predecessors.size(),
      predecessors.begin(), predecessors.begin(),
      [invalid] __device__(vertex_t external, vertex_t internal) -> vertex_t {
        return external < 0 ? invalid : (internal == invalid ? vertex_t{-2} : internal);
      });
    rmm::device_uvector<vertex_t> predecessor_by_vertex(vertex_count, handle_.get_stream());
    rmm::device_uvector<int> present(vertex_count, handle_.get_stream());
    thrust::fill(handle_.get_thrust_policy(), predecessor_by_vertex.begin(),
                 predecessor_by_vertex.end(), invalid);
    thrust::fill(handle_.get_thrust_policy(), present.begin(), present.end(), 0);
    thrust::scatter(handle_.get_thrust_policy(), predecessors.begin(), predecessors.end(),
                    vertices.begin(), predecessor_by_vertex.begin());
    auto const vertex_data = vertices.data();
    auto const present_data = present.data();
    thrust::for_each_n(handle_.get_thrust_policy(), thrust::make_counting_iterator<size_t>(0),
                       vertices.size(), [vertex_data, present_data] __device__(size_t i) -> void {
                         present_data[vertex_data[i]] = 1;
                       });
    auto const source_data = sources.data();
    auto const destination_data = destinations.data();
    auto const predecessor_data = predecessor_by_vertex.data();
    rmm::device_uvector<vertex_t> lengths(destinations.size(), handle_.get_stream());
    // Each destination walk is bounded by the result's vertex count to reject cycles.
    thrust::transform(handle_.get_thrust_policy(), destinations.begin(), destinations.end(),
                      lengths.begin(),
                      [source_data, predecessor_data, present_data, vertex_count, invalid]
                      __device__(vertex_t destination) -> vertex_t {
                        if (destination == invalid) { return vertex_t{-2}; }
                        auto current = destination;
                        for (size_t step = 0; step < static_cast<size_t>(vertex_count); ++step) {
                          if (current == source_data[0]) { return static_cast<vertex_t>(step + 1); }
                          if (present_data[current] == 0) { return vertex_t{-2}; }
                          auto predecessor = predecessor_data[current];
                          if (predecessor == invalid) { return vertex_t{-1}; }
                          if (predecessor == vertex_t{-2}) { return vertex_t{-2}; }
                          current = predecessor;
                        }
                        return vertex_t{-3};
                      });
    auto first_invalid = thrust::find_if(
      handle_.get_thrust_policy(), lengths.begin(), lengths.end(),
      [] __device__(vertex_t length) -> bool { return length < 0; });
    if (first_invalid != lengths.end()) {
      vertex_t status;
      raft::update_host(&status, lengths.data() + (first_invalid - lengths.begin()), 1, handle_.get_stream());
      handle_.sync_stream();
      if (status == vertex_t{-3}) {
        mark_error(CUGRAPH_INVALID_STATE, "cycle detected while reconstructing SSSP predecessor path");
      } else if (status == vertex_t{-2}) {
        mark_error(CUGRAPH_INVALID_INPUT, "destination vertex is not present in the SSSP predecessor result");
      } else {
        mark_error(CUGRAPH_INVALID_INPUT, "destination vertex is unreachable from the SSSP source");
      }
      return;
    }
    auto const max_path_length = thrust::transform_reduce(
      handle_.get_thrust_policy(), lengths.begin(), lengths.end(),
      [] __device__(vertex_t length) -> vertex_t { return length; }, vertex_t{1}, thrust::maximum<vertex_t>{});
    if (destinations.size() > std::numeric_limits<size_t>::max() / max_path_length) {
      mark_error(CUGRAPH_INVALID_STATE, "SSSP path matrix size exceeds addressable storage");
      return;
    }
    rmm::device_uvector<vertex_t> result(destinations.size() * max_path_length, handle_.get_stream());
    thrust::fill(handle_.get_thrust_policy(), result.begin(), result.end(), invalid);
    auto const result_data = result.data();
    auto const length_data = lengths.data();
    thrust::for_each_n(handle_.get_thrust_policy(), thrust::make_counting_iterator<size_t>(0),
                       destinations.size(),
                       [destination_data, predecessor_data, length_data, result_data, max_path_length]
                       __device__(size_t row) -> void {
                         auto current = destination_data[row];
                         auto offset = (row + 1) * max_path_length;
                         for (vertex_t step = 0; step < length_data[row]; ++step) {
                           result_data[--offset] = current;
                           current = predecessor_data[current];
                         }
                       });
    cugraph::unrenumber_int_vertices<vertex_t, false>(
      handle_, result.data(), result.size(), number_map.data(),
      graph_view.vertex_partition_range_lasts(), false);
    auto paths = std::make_unique<cugraph_type_erased_device_array_t>(result, graph_->vertex_type_);
    auto output = std::make_unique<cugraph_extract_paths_result_t>(
      cugraph_extract_paths_result_t{static_cast<size_t>(max_path_length), paths.get()});
    result_ = output.release();
    paths.release();
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
      // BFS and SSSP expect store_transposed == false
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

      if (paths_result_->distances_->type_ == FLOAT32 || paths_result_->distances_->type_ == FLOAT64) {
        if constexpr (multi_gpu) {
          mark_error(CUGRAPH_NOT_IMPLEMENTED, "SSSP path extraction requires a single-GPU graph");
        } else {
          extract_weighted_paths(graph_view, *number_map);
        }
        return;
      }

      rmm::device_uvector<vertex_t> destinations(destinations_->size_, handle_.get_stream());
      raft::copy(destinations.data(),
                 destinations_->as_type<vertex_t>(),
                 destinations_->size_,
                 handle_.get_stream());

      rmm::device_uvector<vertex_t> predecessors(paths_result_->predecessors_->size_,
                                                 handle_.get_stream());
      raft::copy(predecessors.data(),
                 paths_result_->predecessors_->view()->as_type<vertex_t>(),
                 paths_result_->predecessors_->view()->size_,
                 handle_.get_stream());

      //
      // Need to renumber destinations
      //
      renumber_ext_vertices<vertex_t, multi_gpu>(handle_,
                                                 destinations.data(),
                                                 destinations.size(),
                                                 number_map->data(),
                                                 graph_view.local_vertex_partition_range_first(),
                                                 graph_view.local_vertex_partition_range_last(),
                                                 false);

      renumber_ext_vertices<vertex_t, multi_gpu>(handle_,
                                                 predecessors.data(),
                                                 predecessors.size(),
                                                 number_map->data(),
                                                 graph_view.local_vertex_partition_range_first(),
                                                 graph_view.local_vertex_partition_range_last(),
                                                 false);

      auto [result, max_path_length] = cugraph::extract_bfs_paths<vertex_t, edge_t, multi_gpu>(
        handle_,
        graph_view,
        paths_result_->distances_->view()->as_type<vertex_t>(),
        predecessors.data(),
        destinations.data(),
        destinations.size());

      unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                   result.data(),
                                                   result.size(),
                                                   number_map->data(),
                                                   graph_view.vertex_partition_range_lasts(),
                                                   false);

      auto paths = std::make_unique<cugraph_type_erased_device_array_t>(
        result, graph_->vertex_type_);
      auto output = std::make_unique<cugraph_extract_paths_result_t>(
        cugraph_extract_paths_result_t{static_cast<size_t>(max_path_length), paths.get()});
      result_ = output.release();
      paths.release();
    }
  }
};

}  // namespace c_api
}  // namespace cugraph

extern "C" size_t cugraph_extract_paths_result_get_max_path_length(
  cugraph_extract_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_extract_paths_result_t*>(result);
  return internal_pointer->max_path_length_;
}

cugraph_type_erased_device_array_view_t* cugraph_extract_paths_result_get_paths(
  cugraph_extract_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_extract_paths_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->paths_->view());
}

extern "C" void cugraph_extract_paths_result_free(cugraph_extract_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_extract_paths_result_t*>(result);
  delete internal_pointer->paths_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_extract_paths(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* sources,
  const cugraph_paths_result_t* paths_result,
  const cugraph_type_erased_device_array_view_t* destinations,
  cugraph_extract_paths_result_t** result,
  cugraph_error_t** error)
{
  if (error == nullptr) { return CUGRAPH_INVALID_INPUT; }
  *error = nullptr;
  CAPI_EXPECTS(result != nullptr, CUGRAPH_INVALID_INPUT, "result output cannot be null", *error);
  *result = nullptr;
  CAPI_EXPECTS(handle != nullptr, CUGRAPH_INVALID_HANDLE, "handle cannot be null", *error);
  CAPI_EXPECTS(graph != nullptr && sources != nullptr && paths_result != nullptr && destinations != nullptr,
               CUGRAPH_INVALID_INPUT, "path extraction inputs cannot be null", *error);
  auto const* paths = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t const*>(paths_result);
  CAPI_EXPECTS(paths->predecessors_ != nullptr &&
                 (paths->predecessors_->size_ != 0 ||
                  (paths->vertex_ids_ != nullptr && paths->vertex_ids_->size_ == 0)),
               CUGRAPH_INVALID_INPUT, "paths result has no predecessor column", *error);
  CAPI_EXPECTS(paths->vertex_ids_ != nullptr && paths->distances_ != nullptr,
               CUGRAPH_INVALID_STATE, "paths result is missing vertex or distance storage", *error);

  cugraph::c_api::extract_paths_functor functor(handle, graph, sources, paths_result, destinations);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
