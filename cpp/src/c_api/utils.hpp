/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/generic_cascaded_dispatch.hpp"
#include "c_api/graph.hpp"
#include "c_api/error.hpp"

#include <cugraph/utilities/error.hpp>

#include <rmm/error.hpp>

#include <new>
#include <stdexcept>
#include <utility>

// The cuDF shim owns this exception type and installs this public C++ header
// with its other cudf_rust headers.
#include <cudf_rust/error_boundary.hpp>

namespace cugraph {
namespace c_api {

inline cugraph_error_failure_source_t cugraph_failure_source_from_cudf(
  cudf_memory_failure_source_t source) noexcept
{
  switch (source) {
    case CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL:
      return CUGRAPH_ERROR_FAILURE_SOURCE_PHYSICAL;
    case CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT:
      return CUGRAPH_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT;
    default:
      return CUGRAPH_ERROR_FAILURE_SOURCE_UNKNOWN;
  }
}

inline ::cugraph_error_t* make_attributed_allocation_error(
  cudf::rust::memory::attributed_out_of_memory const& ex)
{
  auto const identity = ex.identity();
  return make_allocation_error(ex.what(),
                               CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY,
                               cugraph_failure_source_from_cudf(ex.source()),
                               identity.primary_id,
                               identity.secondary_id,
                               identity.generation,
                               ex.requested_bytes(),
                               ex.alignment(),
                               ex.cuda_error());
}

template <typename operation_t>
cugraph_error_code_t run_with_error_boundary(operation_t&& operation,
                                              ::cugraph_error_t** error)
{
  try {
    return std::forward<operation_t>(operation)();
  } catch (rmm::out_of_memory const& ex) {
    // Catch the owner-attributed derived type before this base-class arm so
    // its limit/physical provenance and domain identity survive the C API.
    if (auto const* attributed = dynamic_cast<
          cudf::rust::memory::attributed_out_of_memory const*>(&ex);
        attributed != nullptr) {
      *error = make_attributed_allocation_error(*attributed);
      return CUGRAPH_ALLOC_ERROR;
    }
    *error = cugraph::c_api::make_allocation_error(
      ex.what(), CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY);
    return CUGRAPH_ALLOC_ERROR;
  } catch (rmm::bad_alloc const& ex) {
    *error = cugraph::c_api::make_allocation_error(
      ex.what(), CUGRAPH_ALLOCATION_SOURCE_RMM_BAD_ALLOC);
    return CUGRAPH_ALLOC_ERROR;
  } catch (std::bad_alloc const& ex) {
    *error = cugraph::c_api::make_allocation_error(
      ex.what(), CUGRAPH_ALLOCATION_SOURCE_STD_BAD_ALLOC);
    return CUGRAPH_ALLOC_ERROR;
  } catch (cugraph::logic_error const& ex) {
    // CUGRAPH_EXPECTS and CUGRAPH_FAIL state a precondition of the call. The reason omits the
    // source location and call stack that what() carries, so the caller can be shown it.
    *error =
      reinterpret_cast<::cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.reason()});
    return CUGRAPH_INVALID_INPUT;
  } catch (std::invalid_argument const& ex) {
    *error = reinterpret_cast<::cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_INVALID_INPUT;
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<::cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

template <typename functor_t, typename result_t>
cugraph_error_code_t run_algorithm(::cugraph_graph_t const* graph,
                                   functor_t& functor,
                                   result_t* result,
                                   ::cugraph_error_t** error)
{
  *result = result_t{};
  *error  = nullptr;

  return run_with_error_boundary(
    [&]() {
      auto p_graph = reinterpret_cast<cugraph::c_api::cugraph_graph_t const*>(graph);

      cugraph::c_api::vertex_dispatcher(p_graph->vertex_type_,
                                        p_graph->edge_type_,
                                        p_graph->weight_type_,
                                        p_graph->edge_type_id_type_,
                                        p_graph->edge_time_type_,
                                        p_graph->store_transposed_,
                                        p_graph->multi_gpu_,
                                        functor);

      if (functor.error_code_ != CUGRAPH_SUCCESS) {
        *error = reinterpret_cast<::cugraph_error_t*>(functor.error_.release());
        return functor.error_code_;
      }

      if constexpr (std::is_same_v<result_t, decltype(functor.result_)>) {
        *result = functor.result_;
      } else {
        *result = reinterpret_cast<result_t>(functor.result_);
      }
      return CUGRAPH_SUCCESS;
    },
    error);
}

}  // namespace c_api
}  // namespace cugraph
