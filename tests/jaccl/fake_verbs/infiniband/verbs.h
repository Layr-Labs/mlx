// Copyright © 2026 Apple Inc.

// A stand-in for <infiniband/verbs.h> for the tests. It declares only the
// types that the JACCL headers use. fake_rdma.cpp defines the three verbs.

#pragma once

#include <cstddef>
#include <cstdint>

struct ibv_device {};
struct ibv_context {};
struct ibv_pd {};
struct ibv_comp_channel {};
struct ibv_qp_init_attr {};
struct ibv_port_attr {};
struct ibv_qp_attr {};
struct ibv_gid {
  uint8_t raw[16];
};

struct ibv_mr {
  uint32_t lkey;
};

struct ibv_cq {
  void* fixture;
};

struct ibv_qp {
  void* fixture;
};

struct ibv_sge {
  uint64_t addr;
  uint32_t length;
  uint32_t lkey;
};

struct ibv_send_wr {
  uint64_t wr_id;
  ibv_send_wr* next;
  ibv_sge* sg_list;
  int num_sge;
  int opcode;
  int send_flags;
};

struct ibv_recv_wr {
  uint64_t wr_id;
  ibv_recv_wr* next;
  ibv_sge* sg_list;
  int num_sge;
};

enum ibv_wc_status { IBV_WC_SUCCESS = 0, IBV_WC_WR_FLUSH_ERR = 5 };
struct ibv_wc {
  uint64_t wr_id;
  ibv_wc_status status;
  uint32_t vendor_err;
  uint32_t byte_len;
};

constexpr int IBV_WR_SEND = 0;
constexpr int IBV_SEND_SIGNALED = 1;

int ibv_post_send(ibv_qp*, ibv_send_wr*, ibv_send_wr**);
int ibv_post_recv(ibv_qp*, ibv_recv_wr*, ibv_recv_wr**);
int ibv_poll_cq(ibv_cq*, int, ibv_wc*);
