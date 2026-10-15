/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph_c/export.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cugraph_error_code_ {
  CUGRAPH_SUCCESS = 0,
  CUGRAPH_UNKNOWN_ERROR,
  CUGRAPH_INVALID_HANDLE,
  CUGRAPH_ALLOC_ERROR,
  CUGRAPH_INVALID_INPUT,
  CUGRAPH_NOT_IMPLEMENTED,
  CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
  CUGRAPH_INVALID_STATE
} cugraph_error_code_t;

typedef enum cugraph_allocation_source_ {
  CUGRAPH_ALLOCATION_SOURCE_UNKNOWN = 0,
  CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY,
  CUGRAPH_ALLOCATION_SOURCE_RMM_BAD_ALLOC,
  CUGRAPH_ALLOCATION_SOURCE_STD_BAD_ALLOC
} cugraph_allocation_source_t;

/**
 * @brief Provenance of an allocation failure, when the allocation owner can
 *        distinguish a logical reservation limit from physical device memory
 *        pressure.
 *
 * This is intentionally separate from `cugraph_allocation_source_t`, which
 * records the C++ exception type caught at the boundary.
 */
typedef enum cugraph_error_failure_source_ {
  CUGRAPH_ERROR_FAILURE_SOURCE_PHYSICAL          = 0,
  CUGRAPH_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT = 1,
  CUGRAPH_ERROR_FAILURE_SOURCE_UNKNOWN           = 2
} cugraph_error_failure_source_t;

typedef struct cugraph_error_ {
  int32_t align_;
} cugraph_error_t;

/**
 * @brief     Return an error message
 *
 * @param [in]  error       The error object from some cugraph function call
 * @return a C-style string that provides detail for the error
 */
CUGRAPH_EXPORT const char* cugraph_error_message(const cugraph_error_t* error);
CUGRAPH_EXPORT cugraph_allocation_source_t cugraph_error_allocation_source(
  const cugraph_error_t* error);
CUGRAPH_EXPORT cugraph_error_failure_source_t cugraph_error_failure_source(
  const cugraph_error_t* error);
CUGRAPH_EXPORT uint64_t cugraph_error_domain_id(const cugraph_error_t* error);
CUGRAPH_EXPORT uint64_t cugraph_error_member_id(const cugraph_error_t* error);
CUGRAPH_EXPORT uint64_t cugraph_error_generation(const cugraph_error_t* error);
CUGRAPH_EXPORT uint64_t cugraph_error_requested_bytes(const cugraph_error_t* error);
CUGRAPH_EXPORT uint64_t cugraph_error_alignment(const cugraph_error_t* error);
CUGRAPH_EXPORT int32_t cugraph_error_cuda_error(const cugraph_error_t* error);

/**
 * @brief    Destroy an error message
 *
 * @param [in]  error       The error object from some cugraph function call
 */
CUGRAPH_EXPORT void cugraph_error_free(cugraph_error_t* error);

#include <cugraph_c/export.h>

#ifdef __cplusplus
}
#endif
