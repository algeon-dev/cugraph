/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

#include <raft/core/error.hpp>

#include <cstdio>
#include <string>
#include <utility>

namespace CUGRAPH_EXPORT cugraph {

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * CUGRAPH_EXPECTS and  CUGRAPH_FAIL macros.
 *
 */
struct logic_error : public raft::exception {
  explicit logic_error(char const* const message) : raft::exception(message), reason_(message) {}
  explicit logic_error(std::string const& message) : raft::exception(message), reason_(message) {}
  logic_error(std::string const& message, std::string reason)
    : raft::exception(message), reason_(std::move(reason))
  {
  }

  /**
   * @brief The violated condition as the macro caller described it.
   *
   * what() prefixes the source location and appends the call stack; this text carries neither, so
   * a boundary can hand it to a caller.
   */
  [[nodiscard]] char const* reason() const noexcept { return reason_.c_str(); }

 private:
  std::string reason_;
};

}  // namespace CUGRAPH_EXPORT cugraph

/**
 * @brief Formats @p fmt and its arguments into @p reason without a location prefix.
 *
 * The local names are unusual so they cannot shadow a variable passed as a format argument.
 */
#define CUGRAPH_SET_ERROR_REASON(reason, fmt, ...)                                      \
  do {                                                                                  \
    int const cugraph_error_reason_size = std::snprintf(nullptr, 0, fmt, ##__VA_ARGS__); \
    if (cugraph_error_reason_size > 0) {                                                \
      reason.resize(static_cast<std::size_t>(cugraph_error_reason_size) + 1);           \
      std::snprintf(reason.data(), reason.size(), fmt, ##__VA_ARGS__);                  \
      reason.resize(static_cast<std::size_t>(cugraph_error_reason_size));               \
    }                                                                                   \
  } while (0)

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when a condition is false
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] fmt String literal description of the reason that cond is expected to be true with
 * optinal format tagas
 * @throw cugraph::logic_error if the condition evaluates to false.
 */
#define CUGRAPH_EXPECTS(cond, fmt, ...)                                     \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::string msg{};                                                    \
      SET_ERROR_MSG(msg, "cuGraph failure at ", fmt, ##__VA_ARGS__);        \
      std::string cugraph_error_reason{};                                   \
      CUGRAPH_SET_ERROR_REASON(cugraph_error_reason, fmt, ##__VA_ARGS__);   \
      throw cugraph::logic_error(msg, std::move(cugraph_error_reason));     \
    }                                                                       \
  } while (0)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * @param[in] fmt String literal description of the reason that this code path is erroneous with
 * optinal format tagas
 * @throw always throws cugraph::logic_error
 */
#define CUGRAPH_FAIL(fmt, ...)                                            \
  do {                                                                    \
    std::string msg{};                                                    \
    SET_ERROR_MSG(msg, "cuGraph failure at ", fmt, ##__VA_ARGS__);        \
    std::string cugraph_error_reason{};                                   \
    CUGRAPH_SET_ERROR_REASON(cugraph_error_reason, fmt, ##__VA_ARGS__);   \
    throw cugraph::logic_error(msg, std::move(cugraph_error_reason));     \
  } while (0)
