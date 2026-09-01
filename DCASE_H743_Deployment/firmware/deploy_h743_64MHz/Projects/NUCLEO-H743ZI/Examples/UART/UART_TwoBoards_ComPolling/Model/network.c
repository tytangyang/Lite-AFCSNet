/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-08-01T02:33:26+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x6125f711a667045337f4627df9942e0f"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-08-01T02:33:26+0800"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  logmel_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 16640, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _in_c_in_c_0_in_c_0_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 33792, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 34816, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 69632, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_freq_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_time_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_1_act_Mul_output_0_0_0__stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 69632, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 34816, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Add_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Sub_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_3_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_after_block_activation_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_after_block_activation_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 34816, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 69632, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_freq_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_time_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_1_act_Mul_output_0_0_0__stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 69632, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 34816, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Add_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Sub_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_3_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_after_block_activation_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34816, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_after_block_activation_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 34816, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 69632, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_freq_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69632, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_time_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_1_act_Mul_output_0_0_0__stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18432, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_shortcut_shortcut_0_AveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Add_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Sub_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_3_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_after_block_activation_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_after_block_activation_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18432, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_freq_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_time_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_1_act_Mul_output_0_0_0__stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18432, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16128, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_after_block_activation_Relu_output_0_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_after_block_activation_Relu_output_0_0_conversion_0_0__stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16128, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 34560, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34560, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34560, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_freq_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34560, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_time_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34560, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34560, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 34560, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_1_act_Mul_output_0_0_0__stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 34560, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16128, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Add_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 56, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 56, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 56, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Sub_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 56, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_3_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_after_block_activation_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  _MaxPool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  _Mul_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  _AveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  _Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  _Add_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  _channel_adjust_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_Add_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 56, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_channel_att_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 56, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 56, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_Sub_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 56, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_Mul_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_Mul_3_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_Add_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16128, AI_STATIC)

/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  _detection_head_detection_head_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2880, AI_STATIC)

/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  _detection_head_detection_head_2_GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 10, AI_STATIC)

/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Constant_2_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  _Sub_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  _Sigmoid_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  _in_c_in_c_0_in_c_0_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 72, AI_STATIC)

/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  _in_c_in_c_0_in_c_0_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_freq_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)

/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_freq_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_time_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)

/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_time_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3, AI_STATIC)

/* Array#150 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#151 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#152 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#153 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_freq_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)

/* Array#154 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_time_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)

/* Array#155 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_time_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#156 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#157 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#158 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3, AI_STATIC)

/* Array#159 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#160 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#161 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_freq_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)

/* Array#162 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_time_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)

/* Array#163 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_time_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#164 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#165 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#166 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3, AI_STATIC)

/* Array#167 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#168 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#169 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_freq_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 320, AI_STATIC)

/* Array#170 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_time_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)

/* Array#171 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_time_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#172 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 896, AI_STATIC)

/* Array#173 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#174 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1680, AI_STATIC)

/* Array#175 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 120, AI_STATIC)

/* Array#176 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_freq_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 600, AI_STATIC)

/* Array#177 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_freq_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 120, AI_STATIC)

/* Array#178 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_time_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 360, AI_STATIC)

/* Array#179 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_time_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 120, AI_STATIC)

/* Array#180 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1680, AI_STATIC)

/* Array#181 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#182 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3, AI_STATIC)

/* Array#183 */
AI_ARRAY_OBJ_DECLARE(
  _channel_adjust_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1792, AI_STATIC)

/* Array#184 */
AI_ARRAY_OBJ_DECLARE(
  _channel_adjust_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 56, AI_STATIC)

/* Array#185 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_channel_att_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3, AI_STATIC)

/* Array#186 */
AI_ARRAY_OBJ_DECLARE(
  _detection_head_detection_head_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 560, AI_STATIC)

/* Array#187 */
AI_ARRAY_OBJ_DECLARE(
  _detection_head_detection_head_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 10, AI_STATIC)

/* Array#188 */
AI_ARRAY_OBJ_DECLARE(
  _in_c_in_c_0_in_c_0_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 292, AI_STATIC)

/* Array#189 */
AI_ARRAY_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5344, AI_STATIC)

/* Array#190 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_freq_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1217, AI_STATIC)

/* Array#191 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_time_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1217, AI_STATIC)

/* Array#192 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 22, AI_STATIC)

/* Array#193 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_freq_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1217, AI_STATIC)

/* Array#194 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_time_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1217, AI_STATIC)

/* Array#195 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 22, AI_STATIC)

/* Array#196 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_freq_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1217, AI_STATIC)

/* Array#197 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_time_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1217, AI_STATIC)

/* Array#198 */
AI_ARRAY_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 22, AI_STATIC)

/* Array#199 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_freq_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1601, AI_STATIC)

/* Array#200 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_time_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1217, AI_STATIC)

/* Array#201 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_freq_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3001, AI_STATIC)

/* Array#202 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_time_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2281, AI_STATIC)

/* Array#203 */
AI_ARRAY_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 22, AI_STATIC)

/* Array#204 */
AI_ARRAY_OBJ_DECLARE(
  _channel_adjust_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 688, AI_STATIC)

/* Array#205 */
AI_ARRAY_OBJ_DECLARE(
  _aff_fusion_channel_att_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 22, AI_STATIC)

/* Array#206 */
AI_ARRAY_OBJ_DECLARE(
  _detection_head_detection_head_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 324, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Add_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04278819262981415f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_AveragePool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06982806324958801f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_MaxPool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06982806324958801f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Mul_1_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0002998443378601223f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04252749681472778f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Sigmoid_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003904729150235653f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Sub_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.68393635249231e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_aff_fusion_Add_1_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5013142228126526f),
    AI_PACK_INTQ_ZP(-66)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_aff_fusion_Add_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.300949364900589f),
    AI_PACK_INTQ_ZP(-76)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_aff_fusion_Mul_1_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.37926867604255676f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_aff_fusion_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.24518181383609772f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_aff_fusion_Mul_3_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.21823452413082123f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_aff_fusion_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.47895655035972595f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04999730736017227f),
    AI_PACK_INTQ_ZP(-19)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_aff_fusion_channel_att_conv_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13194867968559265f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_aff_fusion_channel_att_conv_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01230737566947937f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568393707275f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_channel_adjust_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12259090691804886f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_channel_adjust_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 56,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009317354299128056f, 0.005237096920609474f, 0.0039880480617284775f, 0.006938972044736147f, 0.004347510635852814f, 0.006317291408777237f, 0.005635368637740612f, 0.007513176184147596f, 0.005864191334694624f, 0.006867917720228434f, 0.004527909681200981f, 0.00464818999171257f, 0.00539954612031579f, 0.007702338043600321f, 0.006119131576269865f, 0.006458619609475136f, 0.004224799107760191f, 0.007861212827265263f, 0.004979337099939585f, 0.0048867519944906235f, 0.006297940853983164f, 0.006254294421523809f, 0.005305590573698282f, 0.004323870874941349f, 0.008453934453427792f, 0.006161291152238846f, 0.004860003478825092f, 0.006672116462141275f, 0.0034595965407788754f, 0.007811842020601034f, 0.005483012180775404f, 0.00511753698810935f, 0.005394823383539915f, 0.00423590000718832f, 0.006067936774343252f, 0.005971090402454138f, 0.005468505434691906f, 0.006196075584739447f, 0.006600750144571066f, 0.005308481398969889f, 0.004027728457003832f, 0.007808634079992771f, 0.0043305205181241035f, 0.00425445893779397f, 0.006875726860016584f, 0.005963739939033985f, 0.004066414199769497f, 0.006986585911363363f, 0.006778733339160681f, 0.00605468126013875f, 0.005835549905896187f, 0.0042516677640378475f, 0.0055688186548650265f, 0.004898452199995518f, 0.006432667840272188f, 0.005782182328402996f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_detection_head_detection_head_0_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.24111443758010864f),
    AI_PACK_INTQ_ZP(-22)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_detection_head_detection_head_0_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 10,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002355471719056368f, 0.002462679287418723f, 0.0021141141187399626f, 0.0040204585529863834f, 0.00269701867364347f, 0.002063286490738392f, 0.002885703230276704f, 0.0023332880809903145f, 0.0024552305694669485f, 0.002743310295045376f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_detection_head_detection_head_2_GlobalAveragePool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04616636410355568f),
    AI_PACK_INTQ_ZP(-33)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_in_c_in_c_0_in_c_0_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07213295251131058f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_in_c_in_c_0_in_c_0_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0009605278028175235f, 0.01059176865965128f, 0.0036790852900594473f, 0.006317422725260258f, 0.0031372455414384604f, 0.0034273189958184958f, 0.004687298089265823f, 0.0026670179795473814f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_in_c_in_c_1_in_c_1_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049947649240493774f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_in_c_in_c_1_in_c_1_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0022987143602222204f, 0.0013178772060200572f, 0.0018021612195298076f, 0.002228751312941313f, 0.004346531815826893f, 0.0019366878550499678f, 0.0014458404621109366f, 0.0007039150223135948f, 0.0009267957648262382f, 0.0021922928281128407f, 0.0013378108851611614f, 0.0008331589051522315f, 0.0007283666054718196f, 0.0015660454519093037f, 0.0018156598089262843f, 0.003171299584209919f, 0.0020171531941741705f, 0.0020409543067216873f, 0.0024687661789357662f, 0.0020437086932361126f, 0.001759975217282772f, 0.0024289449211210012f, 0.0014553060755133629f, 0.002765930024906993f, 0.003084203228354454f, 0.003206745022907853f, 0.0016777529381215572f, 0.0050893365405499935f, 0.0013218451058492064f, 0.002809163648635149f, 0.0016684086294844747f, 0.00411056587472558f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_after_block_activation_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06528233736753464f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07035133987665176f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07035133987665176f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_block_block_1_act_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12463120371103287f),
    AI_PACK_INTQ_ZP(-126)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_block_block_1_act_Sigmoid_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_block_block_1_dw_freq_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12424470484256744f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_block_block_1_dw_freq_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004378755111247301f, 0.006541362032294273f, 0.007233010604977608f, 0.0032899973448365927f, 0.0035179960541427135f, 0.005870624911040068f, 0.008113883435726166f, 0.00374620477668941f, 0.005602085031569004f, 0.004217943642288446f, 0.009707583114504814f, 0.004936039913445711f, 0.005404235329478979f, 0.004212402272969484f, 0.005421334877610207f, 0.005994347855448723f, 0.004619682673364878f, 0.005220451857894659f, 0.0029482608661055565f, 0.0052080340683460236f, 0.0041333651170134544f, 0.0036883829161524773f, 0.00611652247607708f, 0.011689765378832817f, 0.003728518495336175f, 0.0028741382993757725f, 0.0038636091630905867f, 0.0035064092371612787f, 0.004045669920742512f, 0.0037654631305485964f, 0.008350134827196598f, 0.0039605749770998955f, 0.0040967464447021484f, 0.0060240584425628185f, 0.006151776295155287f, 0.004778174217790365f, 0.005602315533906221f, 0.004010351840406656f, 0.005302296951413155f, 0.006117424461990595f, 0.006089892238378525f, 0.006879183929413557f, 0.005939537659287453f, 0.004493348766118288f, 0.011809256859123707f, 0.006964779458940029f, 0.007138205226510763f, 0.006395076401531696f, 0.004469648003578186f, 0.004644549451768398f, 0.004346559289842844f, 0.004958148580044508f, 0.0031853769905865192f, 0.0043100276961922646f, 0.0036193872801959515f, 0.006598716601729393f, 0.003957604058086872f, 0.0030726578552275896f, 0.0035325107164680958f, 0.0060286452062428f, 0.004118312615901232f, 0.005911705084145069f, 0.010737255215644836f, 0.005363902077078819f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_block_block_1_dw_time_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.198826864361763f),
    AI_PACK_INTQ_ZP(-31)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_block_block_1_dw_time_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007661681622266769f, 0.038526516407728195f, 0.006265921983867884f, 0.02440357208251953f, 0.012207264080643654f, 0.009053784422576427f, 0.007250561844557524f, 0.04030906781554222f, 0.012747234664857388f, 0.03307180479168892f, 0.021843019872903824f, 0.03497275337576866f, 0.007815128192305565f, 0.07797674834728241f, 0.009876270778477192f, 0.05172527953982353f, 0.03341444954276085f, 0.019374875351786613f, 0.016493365168571472f, 0.006895103026181459f, 0.025568505749106407f, 0.011152696795761585f, 0.005217872094362974f, 0.0072773657739162445f, 0.0238896943628788f, 0.008379328064620495f, 0.034715842455625534f, 0.02179805189371109f, 0.02221280336380005f, 0.03265004605054855f, 0.008191138505935669f, 0.007998581975698471f, 0.0313304141163826f, 0.01852945238351822f, 0.01577533408999443f, 0.016805849969387054f, 0.003991906065493822f, 0.022793015465140343f, 0.011671093292534351f, 0.016014643013477325f, 0.006680825725197792f, 0.0076002441346645355f, 0.008007736876606941f, 0.0053758141584694386f, 0.01509200967848301f, 0.004712001420557499f, 0.004224738571792841f, 0.01812179945409298f, 0.012599781155586243f, 0.016559874638915062f, 0.015432844869792461f, 0.029552651569247246f, 0.00883365236222744f, 0.00627391692250967f, 0.00934574380517006f, 0.019282041117548943f, 0.013713848777115345f, 0.022809568792581558f, 0.0094860615208745f, 0.007213981822133064f, 0.01511513814330101f, 0.008072844706475735f, 0.005155020393431187f, 0.009654403664171696f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15676721930503845f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15676721930503845f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_fuse_mode_Add_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15676721930503845f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_fuse_mode_Mul_1_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12109152972698212f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_fuse_mode_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09989529848098755f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_fuse_mode_Mul_3_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05091560259461403f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_fuse_mode_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.3135344386100769f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0035296203568577766f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020829373970627785f),
    AI_PACK_INTQ_ZP(-72)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.011102897115051746f),
    AI_PACK_INTQ_ZP(70)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003255341434851289f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0025618241634219885f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_after_block_activation_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05735425278544426f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12404907494783401f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12404907494783401f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_block_block_1_act_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07988617569208145f),
    AI_PACK_INTQ_ZP(-125)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_block_block_1_act_Sigmoid_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_block_block_1_dw_freq_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14971649646759033f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_block_block_1_dw_freq_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004344196058809757f, 0.004188521299511194f, 0.0057860324159264565f, 0.004266937263309956f, 0.0038313220720738173f, 0.0032056320924311876f, 0.007713006809353828f, 0.0037486033979803324f, 0.0029833100270479918f, 0.005342987831681967f, 0.006435423158109188f, 0.005532082170248032f, 0.00468713603913784f, 0.004183223005384207f, 0.003923386801034212f, 0.005003185477107763f, 0.0037936216685920954f, 0.003309547668322921f, 0.004466859623789787f, 0.004065470304340124f, 0.003885278943926096f, 0.0047121369279921055f, 0.006756091956049204f, 0.008233187720179558f, 0.004068594891577959f, 0.006222648546099663f, 0.005068941507488489f, 0.007138665299862623f, 0.004748184699565172f, 0.003842772450298071f, 0.007768314331769943f, 0.00782656017690897f, 0.004968097899109125f, 0.0032378872856497765f, 0.007876379415392876f, 0.003031225176528096f, 0.0036957724951207638f, 0.004292836878448725f, 0.0036767697893083096f, 0.003436757018789649f, 0.00374642014503479f, 0.004577136132866144f, 0.006780494470149279f, 0.006512585561722517f, 0.0032298427540808916f, 0.004366470035165548f, 0.004200522322207689f, 0.005763153079897165f, 0.003908509388566017f, 0.004638071171939373f, 0.007971140556037426f, 0.004665009677410126f, 0.004018791951239109f, 0.00575407687574625f, 0.005418081302195787f, 0.003988412208855152f, 0.004142721649259329f, 0.004023660905659199f, 0.008637631312012672f, 0.003687154734507203f, 0.004083333536982536f, 0.0056875888258218765f, 0.005040908697992563f, 0.005717892199754715f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_block_block_1_dw_time_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1616450846195221f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_block_block_1_dw_time_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015948427841067314f, 0.014170912094414234f, 0.00703849270939827f, 0.004300316330045462f, 0.008522003889083862f, 0.008447355590760708f, 0.006238434463739395f, 0.003513842821121216f, 0.03448032587766647f, 0.008179109543561935f, 0.005968490149825811f, 0.02250310406088829f, 0.009792828932404518f, 0.002320508472621441f, 0.00603119982406497f, 0.013669987209141254f, 0.006835324224084616f, 0.00727516645565629f, 0.021754154935479164f, 0.007199403364211321f, 0.04672597721219063f, 0.008598772808909416f, 0.008918100036680698f, 0.009881967678666115f, 0.02105647511780262f, 0.01352360937744379f, 0.006536333356052637f, 0.006103582214564085f, 0.009315053932368755f, 0.009006809443235397f, 0.01371034700423479f, 0.011395356617867947f, 0.007548283785581589f, 0.02022393047809601f, 0.005989087279886007f, 0.00798199325799942f, 0.018217023462057114f, 0.12067393958568573f, 0.030724365264177322f, 0.020022036507725716f, 0.017179295420646667f, 0.006853890605270863f, 0.016730573028326035f, 0.009037094190716743f, 0.009557594545185566f, 0.037266939878463745f, 0.007547379471361637f, 0.008370081894099712f, 0.03187049552798271f, 0.008440052159130573f, 0.005388913210481405f, 0.010142439976334572f, 0.0183185376226902f, 0.010848447680473328f, 0.03894960507750511f, 0.031132416799664497f, 0.008433250710368156f, 0.010370953008532524f, 0.008515208028256893f, 0.06292235106229782f, 0.03152218088507652f, 0.0036342227831482887f, 0.005533930845558643f, 0.0350784957408905f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06732431799173355f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06732431799173355f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_fuse_mode_Add_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09753160923719406f),
    AI_PACK_INTQ_ZP(-43)))

/* Int quant #61 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_fuse_mode_Mul_1_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1106896921992302f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #62 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_fuse_mode_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13056467473506927f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #63 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_fuse_mode_Mul_3_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04820975288748741f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #64 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_fuse_mode_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1346486359834671f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #65 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002228322671726346f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #66 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013594074174761772f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #67 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.016594434157013893f),
    AI_PACK_INTQ_ZP(-111)))

/* Int quant #68 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0066254897974431515f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #69 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0038479920476675034f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #70 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_after_block_activation_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06982806324958801f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #71 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07033657282590866f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #72 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07033657282590866f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #73 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_block_block_1_act_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08646557480096817f),
    AI_PACK_INTQ_ZP(-125)))

/* Int quant #74 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_block_block_1_act_Sigmoid_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #75 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_block_block_1_dw_freq_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12603335082530975f),
    AI_PACK_INTQ_ZP(-14)))

/* Int quant #76 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_block_block_1_dw_freq_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00658697122707963f, 0.005858819931745529f, 0.0038238822016865015f, 0.004460398573428392f, 0.003849348518997431f, 0.005472927819937468f, 0.005572769325226545f, 0.004839075263589621f, 0.004325438290834427f, 0.003984504379332066f, 0.005531399045139551f, 0.005911373067647219f, 0.0034183382522314787f, 0.003354131244122982f, 0.006015608552843332f, 0.003484251443296671f, 0.004452087450772524f, 0.0071141342632472515f, 0.00509883277118206f, 0.004676228854805231f, 0.003361830487847328f, 0.005391356069594622f, 0.00475703738629818f, 0.005150924902409315f, 0.0035829790867865086f, 0.003974307328462601f, 0.0047851437702775f, 0.004892559256404638f, 0.0042220354080200195f, 0.003140998538583517f, 0.006462378427386284f, 0.005502643063664436f, 0.005197191145271063f, 0.003483724081888795f, 0.004183222074061632f, 0.009280352853238583f, 0.005592647939920425f, 0.003971519879996777f, 0.006302704103291035f, 0.003962459973990917f, 0.004519544076174498f, 0.003524844301864505f, 0.003806930733844638f, 0.003696008585393429f, 0.0061320229433476925f, 0.004356378223747015f, 0.004206725861877203f, 0.004847369622439146f, 0.0029031471349298954f, 0.006812334060668945f, 0.005662925075739622f, 0.0033439286053180695f, 0.004375625401735306f, 0.005842478014528751f, 0.004758186638355255f, 0.004831657279282808f, 0.004621104337275028f, 0.006221544463187456f, 0.004912169184535742f, 0.005779427010565996f, 0.004696293734014034f, 0.004885607864707708f, 0.004309311043471098f, 0.005342081654816866f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #77 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_block_block_1_dw_time_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1482931673526764f),
    AI_PACK_INTQ_ZP(-20)))

/* Int quant #78 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_block_block_1_dw_time_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005002033896744251f, 0.009574097581207752f, 0.009030929766595364f, 0.009782891720533371f, 0.01949148438870907f, 0.00550943985581398f, 0.011419578455388546f, 0.0053230770863592625f, 0.004583157133311033f, 0.009355773217976093f, 0.004259803798049688f, 0.011588867753744125f, 0.0066545396111905575f, 0.03825978934764862f, 0.01353618036955595f, 0.007317951414734125f, 0.004204642027616501f, 0.015518482774496078f, 0.013926055282354355f, 0.005088340491056442f, 0.007445772178471088f, 0.020914649590849876f, 0.012018791399896145f, 0.009265033528208733f, 0.003929333761334419f, 0.008115224540233612f, 0.004627411253750324f, 0.01689661294221878f, 0.006225189659744501f, 0.016214555129408836f, 0.006865658797323704f, 0.005460005719214678f, 0.010445619001984596f, 0.00829221773892641f, 0.017326219007372856f, 0.00267981830984354f, 0.01086105965077877f, 0.008744586259126663f, 0.011273764073848724f, 0.011106742545962334f, 0.014814593829214573f, 0.014482516795396805f, 0.008382006548345089f, 0.00823317002505064f, 0.007141308858990669f, 0.010046331211924553f, 0.008763979189097881f, 0.007910278625786304f, 0.010530213825404644f, 0.006022579036653042f, 0.00923685822635889f, 0.023159651085734367f, 0.006952208932489157f, 0.0041489312425255775f, 0.006130795460194349f, 0.012363428249955177f, 0.013462304137647152f, 0.0184037946164608f, 0.018199075013399124f, 0.011902854777872562f, 0.022509044036269188f, 0.012089630588889122f, 0.021873485296964645f, 0.018053432926535606f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #79 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.17416159808635712f),
    AI_PACK_INTQ_ZP(62)))

/* Int quant #80 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.17416159808635712f),
    AI_PACK_INTQ_ZP(62)))

/* Int quant #81 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_fuse_mode_Add_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1983523815870285f),
    AI_PACK_INTQ_ZP(31)))

/* Int quant #82 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_fuse_mode_Mul_1_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11140108108520508f),
    AI_PACK_INTQ_ZP(54)))

/* Int quant #83 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_fuse_mode_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0692932978272438f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #84 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_fuse_mode_Mul_3_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04910671338438988f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #85 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_fuse_mode_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.34832319617271423f),
    AI_PACK_INTQ_ZP(62)))

/* Int quant #86 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00328464200720191f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #87 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009783084504306316f),
    AI_PACK_INTQ_ZP(-49)))

/* Int quant #88 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008950947783887386f),
    AI_PACK_INTQ_ZP(55)))

/* Int quant #89 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007169736549258232f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #90 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002569550881162286f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #91 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage1_b3_shortcut_shortcut_0_AveragePool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05735425278544426f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #92 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b4_after_block_activation_Relu_output_0_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05703173205256462f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #93 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08788832277059555f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #94 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08788832277059555f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #95 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b4_block_block_1_act_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0576423704624176f),
    AI_PACK_INTQ_ZP(-123)))

/* Int quant #96 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b4_block_block_1_act_Sigmoid_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921566531062126f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #97 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b4_block_block_1_dw_freq_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1448625922203064f),
    AI_PACK_INTQ_ZP(-20)))

/* Int quant #98 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b4_block_block_1_dw_freq_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006032587960362434f, 0.007426988333463669f, 0.004494181368499994f, 0.0047710794024169445f, 0.007712814025580883f, 0.007350207772105932f, 0.004723784979432821f, 0.0050097317434847355f, 0.004910038318485022f, 0.005642160307615995f, 0.006774971727281809f, 0.003994304221123457f, 0.004356822930276394f, 0.01078528631478548f, 0.004730349872261286f, 0.0052058761939406395f, 0.006255358923226595f, 0.005378236528486013f, 0.0055678850039839745f, 0.00554021168500185f, 0.00630553113296628f, 0.005865993443876505f, 0.004323278553783894f, 0.006220536306500435f, 0.007258860860019922f, 0.0068232594057917595f, 0.0050964741967618465f, 0.007737893145531416f, 0.005349761340767145f, 0.0058830976486206055f, 0.00672800000756979f, 0.006232450716197491f, 0.006362052168697119f, 0.003991840407252312f, 0.006038597319275141f, 0.005797546822577715f, 0.006361516658216715f, 0.006813128478825092f, 0.0042668855749070644f, 0.005646881647408009f, 0.006095997057855129f, 0.006383063271641731f, 0.004832217935472727f, 0.005651263054460287f, 0.004431054461747408f, 0.006797075737267733f, 0.005097568035125732f, 0.005761891137808561f, 0.005205949302762747f, 0.005565308965742588f, 0.004490642808377743f, 0.00440533272922039f, 0.005378940142691135f, 0.009941736236214638f, 0.0070436024107038975f, 0.004579533357173204f, 0.0045151920057833195f, 0.007363523356616497f, 0.004022900015115738f, 0.007095104083418846f, 0.004412143025547266f, 0.005645420867949724f, 0.005428088363260031f, 0.005682273767888546f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #99 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b4_block_block_1_dw_time_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11099999397993088f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #100 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b4_block_block_1_dw_time_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005354379303753376f, 0.004840129055082798f, 0.004053853917866945f, 0.0040673487819731236f, 0.0044402931816875935f, 0.006763809360563755f, 0.057417258620262146f, 0.002472514985129237f, 0.002836763858795166f, 0.004135883413255215f, 0.008572466671466827f, 0.0071892524138092995f, 0.027563123032450676f, 0.009991145692765713f, 0.0036131602246314287f, 0.007737180683761835f, 0.006951889023184776f, 0.00255193910561502f, 0.008190780878067017f, 0.004176500719040632f, 0.007048511877655983f, 0.0030603772029280663f, 0.014104442670941353f, 0.0044473824091255665f, 0.0053167627193033695f, 0.005013603717088699f, 0.017251305282115936f, 0.0036558008287101984f, 0.030505992472171783f, 0.005858871154487133f, 0.0068131303414702415f, 0.005383442156016827f, 0.005173447076231241f, 0.006970007438212633f, 0.010017902590334415f, 0.008423274382948875f, 0.010835825465619564f, 0.004199420101940632f, 0.014144722372293472f, 0.002719949232414365f, 0.008988231420516968f, 0.00331478682346642f, 0.0043978760950267315f, 0.007159465458244085f, 0.006207956932485104f, 0.001787487301044166f, 0.0031937207095324993f, 0.002564058406278491f, 0.006586242001503706f, 0.0067085628397762775f, 0.0048271589912474155f, 0.0062038288451731205f, 0.0040632798336446285f, 0.009390723891556263f, 0.0022663886193186045f, 0.012985337525606155f, 0.011835905723273754f, 0.013236607424914837f, 0.0239885114133358f, 0.005828353576362133f, 0.02091405540704727f, 0.0026411451399326324f, 0.019959328696131706f, 0.001795900403521955f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #101 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10564198344945908f),
    AI_PACK_INTQ_ZP(-11)))

/* Int quant #102 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10564198344945908f),
    AI_PACK_INTQ_ZP(-11)))

/* Int quant #103 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_after_block_activation_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.23947827517986298f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #104 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1260804384946823f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #105 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1260804384946823f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #106 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_block_block_1_act_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06332184374332428f),
    AI_PACK_INTQ_ZP(-124)))

/* Int quant #107 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_block_block_1_act_Sigmoid_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921567928045988f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #108 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_block_block_1_dw_freq_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.17456591129302979f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #109 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_block_block_1_dw_freq_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 120,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005131647922098637f, 0.005966132506728172f, 0.004044347442686558f, 0.005015934351831675f, 0.0045271702110767365f, 0.004914097022265196f, 0.004057917278259993f, 0.006688637658953667f, 0.005758686922490597f, 0.005202538799494505f, 0.007177221588790417f, 0.005543423350900412f, 0.005516876466572285f, 0.0038605458103120327f, 0.005039538722485304f, 0.005796344950795174f, 0.007882226258516312f, 0.004585735034197569f, 0.005550163332372904f, 0.005146813113242388f, 0.005268948618322611f, 0.005799660459160805f, 0.004171502310782671f, 0.005745661444962025f, 0.004368296358734369f, 0.005335419904440641f, 0.0071516516618430614f, 0.005912665743380785f, 0.004992366768419743f, 0.004282718058675528f, 0.006442005280405283f, 0.00510791689157486f, 0.005604972131550312f, 0.004210513085126877f, 0.0056613958440721035f, 0.006180851720273495f, 0.005975479260087013f, 0.00030162528855726123f, 0.00604290422052145f, 0.0066629513166844845f, 0.007472934667021036f, 0.00369059550575912f, 0.005700389388948679f, 0.005819336976855993f, 0.0043367440812289715f, 0.005218867212533951f, 0.007838430814445019f, 0.005306003615260124f, 0.005530299618840218f, 0.005083777010440826f, 0.004036920610815287f, 0.005235418677330017f, 0.005248958710581064f, 0.004109066911041737f, 0.006765023339539766f, 0.004685360472649336f, 0.004503671079874039f, 0.004491504747420549f, 0.004542069509625435f, 0.0056841932237148285f, 0.009918969124555588f, 0.0058823563158512115f, 0.005143402144312859f, 0.004972175229340792f, 0.005939044523984194f, 0.005134009290486574f, 0.00924121867865324f, 0.005849271547049284f, 0.007732196245342493f, 0.003537424374371767f, 0.007278625387698412f, 0.0044112298637628555f, 0.00981076993048191f, 0.006522445939481258f, 0.004766545724123716f, 0.004481002222746611f, 0.004096293356269598f, 0.005604917649179697f, 0.0044725388288497925f, 0.0063288393430411816f, 0.004949592053890228f, 0.005545030813664198f, 0.004754005931317806f, 0.0038886521942913532f, 0.0052562919445335865f, 0.005025767255574465f, 0.005021396093070507f, 0.004437639843672514f, 0.005660045892000198f, 0.004037223756313324f, 0.004814510699361563f, 0.0068466621451079845f, 0.006683040875941515f, 0.0056958915665745735f, 0.008074280805885792f, 0.003781517967581749f, 0.005960402544587851f, 0.0051488569006323814f, 0.004535405430942774f, 0.00692063057795167f, 0.004747102037072182f, 0.006986632943153381f, 0.006435913499444723f, 0.004587708972394466f, 0.004754949361085892f, 0.004759090021252632f, 0.00419003376737237f, 0.010729409754276276f, 0.01201638113707304f, 0.010543445125222206f, 0.009610847570002079f, 0.005700966343283653f, 0.008357984013855457f, 0.004597777500748634f, 0.00713225407525897f, 0.005599531810730696f, 0.0037699018139392138f, 0.0052559818141162395f, 0.0036663750652223825f, 0.007481452077627182f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #110 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_block_block_1_dw_time_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1329929530620575f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #111 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_block_block_1_dw_time_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 120,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012956162914633751f, 0.012008918449282646f, 0.037118613719940186f, 0.005697723478078842f, 0.009913594461977482f, 0.006703604478389025f, 0.007140916306525469f, 0.006047571077942848f, 0.0036943641025573015f, 0.00449346611276269f, 0.007677923422306776f, 0.002578080166131258f, 0.009094557724893093f, 0.006135411094874144f, 0.00650201877579093f, 0.012409542687237263f, 0.005666297394782305f, 0.01010709535330534f, 0.011246505193412304f, 0.005710639525204897f, 0.020634949207305908f, 0.02128235436975956f, 0.01695612445473671f, 0.024517739191651344f, 0.008344735950231552f, 0.008056988008320332f, 0.00877078901976347f, 0.006462229415774345f, 0.021988078951835632f, 0.06149320676922798f, 0.00375870568677783f, 0.002923180116340518f, 0.005266761407256126f, 0.012578802183270454f, 0.006234730593860149f, 0.012286593206226826f, 0.015469674952328205f, 0.09936530888080597f, 0.006888833828270435f, 0.0034267015289515257f, 0.0035555886570364237f, 0.06962636113166809f, 0.004350773524492979f, 0.009023561142385006f, 0.008708326146006584f, 0.012533190660178661f, 0.0037840004079043865f, 0.00493325712159276f, 0.01668405532836914f, 0.007055678870528936f, 0.014436734840273857f, 0.08986551314592361f, 0.0057418025098741055f, 0.011906319297850132f, 0.008902953006327152f, 0.025962119922041893f, 0.006378122139722109f, 0.009012438356876373f, 0.006455589085817337f, 0.009724373929202557f, 0.03148231282830238f, 0.008713288232684135f, 0.0029116610530763865f, 0.017268581315875053f, 0.00497085927054286f, 0.012715501710772514f, 0.007268303539603949f, 0.024825096130371094f, 0.006521864794194698f, 0.0029094817582517862f, 0.016143208369612694f, 0.01985001191496849f, 0.0073882886208593845f, 0.00605128426104784f, 0.04463912174105644f, 0.005765580106526613f, 0.015213781967759132f, 0.017772532999515533f, 0.0065483208745718f, 0.004056205041706562f, 0.028006134554743767f, 0.010990384966135025f, 0.023644346743822098f, 0.005713040940463543f, 0.003690567333251238f, 0.018022768199443817f, 0.013738161884248257f, 0.008877231739461422f, 0.0053693088702857494f, 0.008667564950883389f, 0.014778843149542809f, 0.009878518059849739f, 0.010970366187393665f, 0.005661847535520792f, 0.017997613176703453f, 0.024051645770668983f, 0.006972881965339184f, 0.005783890374004841f, 0.006275464314967394f, 0.005914479028433561f, 0.008504797704517841f, 0.011992577463388443f, 0.006383577361702919f, 0.0021993941627442837f, 0.0305545162409544f, 0.023444412276148796f, 0.011616760864853859f, 0.006489025894552469f, 0.002732026856392622f, 0.0036670081317424774f, 0.004807208199054003f, 0.011744292452931404f, 0.0025467246305197477f, 0.012025953270494938f, 0.003905985504388809f, 0.02219739742577076f, 0.04887141287326813f, 0.013255879282951355f, 0.007443299517035484f, 0.013708067126572132f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #112 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2670440673828125f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #113 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2670440673828125f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #114 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_fuse_mode_Add_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.26969215273857117f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #115 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_fuse_mode_Mul_1_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.48317286372184753f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #116 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_fuse_mode_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11406346410512924f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #117 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_fuse_mode_Mul_3_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07745815813541412f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #118 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_fuse_mode_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.534088134765625f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #119 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0039018606767058372f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #120 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027043357491493225f),
    AI_PACK_INTQ_ZP(-44)))

/* Int quant #121 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06913188844919205f),
    AI_PACK_INTQ_ZP(-52)))

/* Int quant #122 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.019202280789613724f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #123 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921551164239645f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #124 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logmel_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09422463923692703f),
    AI_PACK_INTQ_ZP(-8)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Add_output_0_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 32), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_Add_output_0_output_array, &_Add_output_0_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _AveragePool_output_0_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 32), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_AveragePool_output_0_output_array, &_AveragePool_output_0_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _MaxPool_output_0_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 32), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_MaxPool_output_0_output_array, &_MaxPool_output_0_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _Mul_1_output_0_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 32), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_Mul_1_output_0_output_array, &_Mul_1_output_0_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _Mul_output_0_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 32), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_Mul_output_0_output_array, &_Mul_output_0_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _Sigmoid_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_Sigmoid_output_0_DequantizeLinear_Output_const_4D_array, &_Sigmoid_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _Sub_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_Sub_output_0_DequantizeLinear_Output_const_4D_array, &_Sub_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_Add_1_output_0_output, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_aff_fusion_Add_1_output_0_output_array, &_aff_fusion_Add_1_output_0_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_Add_output_0_output, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_aff_fusion_Add_output_0_output_array, &_aff_fusion_Add_output_0_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_Mul_1_output_0_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_aff_fusion_Mul_1_output_0_output_array, &_aff_fusion_Mul_1_output_0_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_Mul_2_output_0_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_aff_fusion_Mul_2_output_0_output_array, &_aff_fusion_Mul_2_output_0_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_Mul_3_output_0_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_aff_fusion_Mul_3_output_0_output_array, &_aff_fusion_Mul_3_output_0_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_Mul_output_0_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_aff_fusion_Mul_output_0_output_array, &_aff_fusion_Mul_output_0_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 1, 1, 56, 56),
  1, &_aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_output_array, &_aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_Sub_output_0_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &_aff_fusion_Sub_output_0_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 1, 1, 56, 56),
  1, &_aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, &_aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_output0, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 56), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, &_aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_channel_att_conv_Conv_output_0_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 56), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_aff_fusion_channel_att_conv_Conv_output_0_output_array, &_aff_fusion_channel_att_conv_Conv_output_0_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_channel_att_conv_Conv_output_0_scratch0, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 22, 1, 1), AI_STRIDE_INIT(4, 1, 1, 22, 22),
  1, &_aff_fusion_channel_att_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_channel_att_conv_Conv_output_0_weights, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 3, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_aff_fusion_channel_att_conv_Conv_output_0_weights_array, &_aff_fusion_channel_att_conv_Conv_output_0_weights_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_output, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 56), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_output_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_output0, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_output_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 56), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output_array, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output0, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 1, 1, 56, 56),
  1, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output_array, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _channel_adjust_Conv_output_0_bias, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &_channel_adjust_Conv_output_0_bias_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _channel_adjust_Conv_output_0_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_channel_adjust_Conv_output_0_output_array, &_channel_adjust_Conv_output_0_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _channel_adjust_Conv_output_0_scratch0, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 688, 1, 1), AI_STRIDE_INIT(4, 1, 1, 688, 688),
  1, &_channel_adjust_Conv_output_0_scratch0_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _channel_adjust_Conv_output_0_weights, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 32, 1, 1, 56), AI_STRIDE_INIT(4, 1, 32, 1792, 1792),
  1, &_channel_adjust_Conv_output_0_weights_array, &_channel_adjust_Conv_output_0_weights_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  _detection_head_detection_head_0_Conv_output_0_bias, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &_detection_head_detection_head_0_Conv_output_0_bias_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  _detection_head_detection_head_0_Conv_output_0_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 9, 32), AI_STRIDE_INIT(4, 1, 1, 10, 90),
  1, &_detection_head_detection_head_0_Conv_output_0_output_array, &_detection_head_detection_head_0_Conv_output_0_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  _detection_head_detection_head_0_Conv_output_0_scratch0, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 324, 1, 1), AI_STRIDE_INIT(4, 1, 1, 324, 324),
  1, &_detection_head_detection_head_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  _detection_head_detection_head_0_Conv_output_0_weights, AI_STATIC,
  31, 0x1,
  AI_SHAPE_INIT(4, 56, 1, 1, 10), AI_STRIDE_INIT(4, 1, 56, 560, 560),
  1, &_detection_head_detection_head_0_Conv_output_0_weights_array, &_detection_head_detection_head_0_Conv_output_0_weights_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  _detection_head_detection_head_2_GlobalAveragePool_output_0_output, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 1, 1, 10, 10),
  1, &_detection_head_detection_head_2_GlobalAveragePool_output_0_output_array, &_detection_head_detection_head_2_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  _in_c_in_c_0_in_c_0_2_Relu_output_0_bias, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_in_c_in_c_0_in_c_0_2_Relu_output_0_bias_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  _in_c_in_c_0_in_c_0_2_Relu_output_0_output, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 33, 128), AI_STRIDE_INIT(4, 1, 1, 8, 264),
  1, &_in_c_in_c_0_in_c_0_2_Relu_output_0_output_array, &_in_c_in_c_0_in_c_0_2_Relu_output_0_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  _in_c_in_c_0_in_c_0_2_Relu_output_0_scratch0, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 292, 1, 1), AI_STRIDE_INIT(4, 1, 1, 292, 292),
  1, &_in_c_in_c_0_in_c_0_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  _in_c_in_c_0_in_c_0_2_Relu_output_0_weights, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 3, 8), AI_STRIDE_INIT(4, 1, 1, 8, 24),
  1, &_in_c_in_c_0_in_c_0_2_Relu_output_0_weights_array, &_in_c_in_c_0_in_c_0_2_Relu_output_0_weights_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 4, 4, 128, 2176),
  1, &_in_c_in_c_1_in_c_1_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_bias, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_in_c_in_c_1_in_c_1_2_Relu_output_0_bias_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_in_c_in_c_1_in_c_1_2_Relu_output_0_output_array, &_in_c_in_c_1_in_c_1_2_Relu_output_0_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_scratch0, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 5344, 1, 1), AI_STRIDE_INIT(4, 1, 1, 5344, 5344),
  1, &_in_c_in_c_1_in_c_1_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_weights, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 8, 3, 3, 32), AI_STRIDE_INIT(4, 1, 8, 256, 768),
  1, &_in_c_in_c_1_in_c_1_2_Relu_output_0_weights_array, &_in_c_in_c_1_in_c_1_2_Relu_output_0_weights_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_after_block_activation_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 4, 4, 128, 2176),
  1, &_stage1_b1_after_block_activation_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_after_block_activation_Relu_output_0_output, AI_STATIC,
  43, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b1_after_block_activation_Relu_output_0_output_array, &_stage1_b1_after_block_activation_Relu_output_0_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_output, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_output0, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(5, 1, 16, 17, 64, 4), AI_STRIDE_INIT(5, 1, 1, 64, 1088, 16),
  1, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 4, 4, 256, 4352),
  1, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 64), AI_STRIDE_INIT(4, 4, 32, 2048, 2048),
  1, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_output, AI_STATIC,
  49, 0x1,
  AI_SHAPE_INIT(5, 1, 4, 17, 64, 16), AI_STRIDE_INIT(5, 1, 1, 64, 1088, 4),
  1, &_stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_output_array, &_stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_output0, AI_STATIC,
  50, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_output_array, &_stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_1_act_Mul_output_0_0_0__stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 4, 4, 256, 4352),
  1, &_stage1_b1_block_block_1_act_Mul_output_0_0_0__stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_1_act_Mul_output_0_output, AI_STATIC,
  52, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b1_block_block_1_act_Mul_output_0_output_array, &_stage1_b1_block_block_1_act_Mul_output_0_output_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_1_act_Sigmoid_output_0_output, AI_STATIC,
  53, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b1_block_block_1_act_Sigmoid_output_0_output_array, &_stage1_b1_block_block_1_act_Sigmoid_output_0_output_array_intq)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_freq_Conv_output_0_bias, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_bias_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_freq_Conv_output_0_output, AI_STATIC,
  55, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_output_array, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_output_array_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_freq_Conv_output_0_scratch0, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 1217, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1217, 1217),
  1, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_scratch0_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_freq_Conv_output_0_weights, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 3, 1), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_weights_array, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_weights_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_time_Conv_output_0_bias, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_stage1_b1_block_block_1_dw_time_Conv_output_0_bias_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_time_Conv_output_0_output, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b1_block_block_1_dw_time_Conv_output_0_output_array, &_stage1_b1_block_block_1_dw_time_Conv_output_0_output_array_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_time_Conv_output_0_scratch0, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 1217, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1217, 1217),
  1, &_stage1_b1_block_block_1_dw_time_Conv_output_0_scratch0_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_time_Conv_output_0_weights, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 1, 1), AI_STRIDE_INIT(4, 1, 64, 64, 192),
  1, &_stage1_b1_block_block_1_dw_time_Conv_output_0_weights_array, &_stage1_b1_block_block_1_dw_time_Conv_output_0_weights_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_output, AI_STATIC,
  62, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_output0, AI_STATIC,
  63, 0x1,
  AI_SHAPE_INIT(5, 1, 8, 17, 64, 4), AI_STRIDE_INIT(5, 1, 1, 32, 544, 8),
  1, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 4, 4, 128, 2176),
  1, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 2048),
  1, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output, AI_STATIC,
  67, 0x1,
  AI_SHAPE_INIT(5, 1, 4, 17, 64, 8), AI_STRIDE_INIT(5, 1, 1, 32, 544, 4),
  1, &_stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output_array, &_stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output0, AI_STATIC,
  68, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output_array, &_stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Add_output_0_output, AI_STATIC,
  69, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b1_fuse_mode_Add_output_0_output_array, &_stage1_b1_fuse_mode_Add_output_0_output_array_intq)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Constant_2_output_0_4D, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_stage1_b1_fuse_mode_Constant_2_output_0_4D_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  71, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D_array, &_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_1_output_0_output, AI_STATIC,
  72, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b1_fuse_mode_Mul_1_output_0_output_array, &_stage1_b1_fuse_mode_Mul_1_output_0_output_array_intq)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_2_output_0_output, AI_STATIC,
  73, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b1_fuse_mode_Mul_2_output_0_output_array, &_stage1_b1_fuse_mode_Mul_2_output_0_output_array_intq)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_3_output_0_output, AI_STATIC,
  74, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b1_fuse_mode_Mul_3_output_0_output_array, &_stage1_b1_fuse_mode_Mul_3_output_0_output_array_intq)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_output_0_output, AI_STATIC,
  75, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b1_fuse_mode_Mul_output_0_output_array, &_stage1_b1_fuse_mode_Mul_output_0_output_array_intq)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_output, AI_STATIC,
  76, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_output_array, &_stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_output_array_intq)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Sub_output_0_output, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_stage1_b1_fuse_mode_Sub_output_0_output_array, NULL)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output, AI_STATIC,
  78, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, &_stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output0, AI_STATIC,
  79, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, &_stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_bias, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_bias_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_output, AI_STATIC,
  81, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_output_array, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_output_array_intq)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_scratch0, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 1, 22, 1, 1), AI_STRIDE_INIT(4, 1, 1, 22, 22),
  1, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_weights, AI_STATIC,
  83, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 3, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_weights_array, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_weights_array_intq)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_output, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_output_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_output0, AI_STATIC,
  85, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_output_array, NULL)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output, AI_STATIC,
  86, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array_intq)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output0, AI_STATIC,
  87, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array_intq)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_after_block_activation_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output, AI_STATIC,
  88, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 4, 4, 128, 2176),
  1, &_stage1_b2_after_block_activation_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_after_block_activation_Relu_output_0_output, AI_STATIC,
  89, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b2_after_block_activation_Relu_output_0_output_array, &_stage1_b2_after_block_activation_Relu_output_0_output_array_intq)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_output, AI_STATIC,
  90, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_output0, AI_STATIC,
  91, 0x1,
  AI_SHAPE_INIT(5, 1, 16, 17, 64, 4), AI_STRIDE_INIT(5, 1, 1, 64, 1088, 16),
  1, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias, AI_STATIC,
  92, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array, NULL)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output, AI_STATIC,
  93, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 4, 4, 256, 4352),
  1, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array, NULL)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights, AI_STATIC,
  94, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 64), AI_STRIDE_INIT(4, 4, 32, 2048, 2048),
  1, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array, NULL)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_output, AI_STATIC,
  95, 0x1,
  AI_SHAPE_INIT(5, 1, 4, 17, 64, 16), AI_STRIDE_INIT(5, 1, 1, 64, 1088, 4),
  1, &_stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_output_array, &_stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_output0, AI_STATIC,
  96, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_output_array, &_stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_1_act_Mul_output_0_0_0__stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output, AI_STATIC,
  97, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 4, 4, 256, 4352),
  1, &_stage1_b2_block_block_1_act_Mul_output_0_0_0__stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array, NULL)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_1_act_Mul_output_0_output, AI_STATIC,
  98, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b2_block_block_1_act_Mul_output_0_output_array, &_stage1_b2_block_block_1_act_Mul_output_0_output_array_intq)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_1_act_Sigmoid_output_0_output, AI_STATIC,
  99, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b2_block_block_1_act_Sigmoid_output_0_output_array, &_stage1_b2_block_block_1_act_Sigmoid_output_0_output_array_intq)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_freq_Conv_output_0_output, AI_STATIC,
  100, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b2_block_block_1_dw_freq_Conv_output_0_output_array, &_stage1_b2_block_block_1_dw_freq_Conv_output_0_output_array_intq)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_freq_Conv_output_0_scratch0, AI_STATIC,
  101, 0x0,
  AI_SHAPE_INIT(4, 1, 1217, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1217, 1217),
  1, &_stage1_b2_block_block_1_dw_freq_Conv_output_0_scratch0_array, NULL)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_freq_Conv_output_0_weights, AI_STATIC,
  102, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 3, 1), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &_stage1_b2_block_block_1_dw_freq_Conv_output_0_weights_array, &_stage1_b2_block_block_1_dw_freq_Conv_output_0_weights_array_intq)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_time_Conv_output_0_bias, AI_STATIC,
  103, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_stage1_b2_block_block_1_dw_time_Conv_output_0_bias_array, NULL)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_time_Conv_output_0_output, AI_STATIC,
  104, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b2_block_block_1_dw_time_Conv_output_0_output_array, &_stage1_b2_block_block_1_dw_time_Conv_output_0_output_array_intq)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_time_Conv_output_0_scratch0, AI_STATIC,
  105, 0x0,
  AI_SHAPE_INIT(4, 1, 1217, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1217, 1217),
  1, &_stage1_b2_block_block_1_dw_time_Conv_output_0_scratch0_array, NULL)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_time_Conv_output_0_weights, AI_STATIC,
  106, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 1, 1), AI_STRIDE_INIT(4, 1, 64, 64, 192),
  1, &_stage1_b2_block_block_1_dw_time_Conv_output_0_weights_array, &_stage1_b2_block_block_1_dw_time_Conv_output_0_weights_array_intq)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_output, AI_STATIC,
  107, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_output0, AI_STATIC,
  108, 0x1,
  AI_SHAPE_INIT(5, 1, 8, 17, 64, 4), AI_STRIDE_INIT(5, 1, 1, 32, 544, 8),
  1, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias, AI_STATIC,
  109, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array, NULL)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output, AI_STATIC,
  110, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 4, 4, 128, 2176),
  1, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array, NULL)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights, AI_STATIC,
  111, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 2048),
  1, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array, NULL)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output, AI_STATIC,
  112, 0x1,
  AI_SHAPE_INIT(5, 1, 4, 17, 64, 8), AI_STRIDE_INIT(5, 1, 1, 32, 544, 4),
  1, &_stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output_array, &_stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output0, AI_STATIC,
  113, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output_array, &_stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Add_output_0_output, AI_STATIC,
  114, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b2_fuse_mode_Add_output_0_output_array, &_stage1_b2_fuse_mode_Add_output_0_output_array_intq)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_1_output_0_output, AI_STATIC,
  115, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b2_fuse_mode_Mul_1_output_0_output_array, &_stage1_b2_fuse_mode_Mul_1_output_0_output_array_intq)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_2_output_0_output, AI_STATIC,
  116, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b2_fuse_mode_Mul_2_output_0_output_array, &_stage1_b2_fuse_mode_Mul_2_output_0_output_array_intq)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_3_output_0_output, AI_STATIC,
  117, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b2_fuse_mode_Mul_3_output_0_output_array, &_stage1_b2_fuse_mode_Mul_3_output_0_output_array_intq)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_output_0_output, AI_STATIC,
  118, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 17, 64), AI_STRIDE_INIT(4, 1, 1, 32, 544),
  1, &_stage1_b2_fuse_mode_Mul_output_0_output_array, &_stage1_b2_fuse_mode_Mul_output_0_output_array_intq)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_output, AI_STATIC,
  119, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_output_array, &_stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_output_array_intq)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Sub_output_0_output, AI_STATIC,
  120, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_stage1_b2_fuse_mode_Sub_output_0_output_array, NULL)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output, AI_STATIC,
  121, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, &_stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output0, AI_STATIC,
  122, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, &_stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_output, AI_STATIC,
  123, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_output_array, &_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_output_array_intq)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_scratch0, AI_STATIC,
  124, 0x0,
  AI_SHAPE_INIT(4, 1, 22, 1, 1), AI_STRIDE_INIT(4, 1, 1, 22, 22),
  1, &_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_weights, AI_STATIC,
  125, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 3, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_weights_array, &_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_weights_array_intq)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_output, AI_STATIC,
  126, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_output_array, NULL)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_output0, AI_STATIC,
  127, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_output_array, NULL)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output, AI_STATIC,
  128, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array_intq)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output0, AI_STATIC,
  129, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array_intq)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_after_block_activation_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output, AI_STATIC,
  130, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 9, 64), AI_STRIDE_INIT(4, 4, 4, 128, 1152),
  1, &_stage1_b3_after_block_activation_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array, NULL)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_after_block_activation_Relu_output_0_output, AI_STATIC,
  131, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 64), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_stage1_b3_after_block_activation_Relu_output_0_output_array, &_stage1_b3_after_block_activation_Relu_output_0_output_array_intq)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_output, AI_STATIC,
  132, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_output0, AI_STATIC,
  133, 0x1,
  AI_SHAPE_INIT(5, 1, 16, 17, 64, 4), AI_STRIDE_INIT(5, 1, 1, 64, 1088, 16),
  1, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias, AI_STATIC,
  134, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array, NULL)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output, AI_STATIC,
  135, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 4, 4, 256, 4352),
  1, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array, NULL)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights, AI_STATIC,
  136, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 64), AI_STRIDE_INIT(4, 4, 32, 2048, 2048),
  1, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array, NULL)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_output, AI_STATIC,
  137, 0x1,
  AI_SHAPE_INIT(5, 1, 4, 17, 64, 16), AI_STRIDE_INIT(5, 1, 1, 64, 1088, 4),
  1, &_stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_output_array, &_stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_output0, AI_STATIC,
  138, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_output_array, &_stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_1_act_Mul_output_0_0_0__stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output, AI_STATIC,
  139, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 9, 64), AI_STRIDE_INIT(4, 4, 4, 256, 2304),
  1, &_stage1_b3_block_block_1_act_Mul_output_0_0_0__stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array, NULL)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_1_act_Mul_output_0_output, AI_STATIC,
  140, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 9, 64), AI_STRIDE_INIT(4, 1, 1, 64, 576),
  1, &_stage1_b3_block_block_1_act_Mul_output_0_output_array, &_stage1_b3_block_block_1_act_Mul_output_0_output_array_intq)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_1_act_Sigmoid_output_0_output, AI_STATIC,
  141, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 9, 64), AI_STRIDE_INIT(4, 1, 1, 64, 576),
  1, &_stage1_b3_block_block_1_act_Sigmoid_output_0_output_array, &_stage1_b3_block_block_1_act_Sigmoid_output_0_output_array_intq)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_freq_Conv_output_0_output, AI_STATIC,
  142, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 17, 64), AI_STRIDE_INIT(4, 1, 1, 64, 1088),
  1, &_stage1_b3_block_block_1_dw_freq_Conv_output_0_output_array, &_stage1_b3_block_block_1_dw_freq_Conv_output_0_output_array_intq)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_freq_Conv_output_0_scratch0, AI_STATIC,
  143, 0x0,
  AI_SHAPE_INIT(4, 1, 1217, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1217, 1217),
  1, &_stage1_b3_block_block_1_dw_freq_Conv_output_0_scratch0_array, NULL)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_freq_Conv_output_0_weights, AI_STATIC,
  144, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 3, 1), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &_stage1_b3_block_block_1_dw_freq_Conv_output_0_weights_array, &_stage1_b3_block_block_1_dw_freq_Conv_output_0_weights_array_intq)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_time_Conv_output_0_bias, AI_STATIC,
  145, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_stage1_b3_block_block_1_dw_time_Conv_output_0_bias_array, NULL)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_time_Conv_output_0_output, AI_STATIC,
  146, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 9, 64), AI_STRIDE_INIT(4, 1, 1, 64, 576),
  1, &_stage1_b3_block_block_1_dw_time_Conv_output_0_output_array, &_stage1_b3_block_block_1_dw_time_Conv_output_0_output_array_intq)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_time_Conv_output_0_scratch0, AI_STATIC,
  147, 0x0,
  AI_SHAPE_INIT(4, 1, 1217, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1217, 1217),
  1, &_stage1_b3_block_block_1_dw_time_Conv_output_0_scratch0_array, NULL)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_time_Conv_output_0_weights, AI_STATIC,
  148, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 1, 1), AI_STRIDE_INIT(4, 1, 64, 64, 192),
  1, &_stage1_b3_block_block_1_dw_time_Conv_output_0_weights_array, &_stage1_b3_block_block_1_dw_time_Conv_output_0_weights_array_intq)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_output, AI_STATIC,
  149, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 64), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_output0, AI_STATIC,
  150, 0x1,
  AI_SHAPE_INIT(5, 1, 8, 9, 64, 4), AI_STRIDE_INIT(5, 1, 1, 32, 288, 8),
  1, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #151 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias, AI_STATIC,
  151, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array, NULL)

/* Tensor #152 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output, AI_STATIC,
  152, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 9, 64), AI_STRIDE_INIT(4, 4, 4, 128, 1152),
  1, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array, NULL)

/* Tensor #153 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights, AI_STATIC,
  153, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 2048),
  1, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array, NULL)

/* Tensor #154 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output, AI_STATIC,
  154, 0x1,
  AI_SHAPE_INIT(5, 1, 4, 9, 64, 8), AI_STRIDE_INIT(5, 1, 1, 32, 288, 4),
  1, &_stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output_array, &_stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #155 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output0, AI_STATIC,
  155, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 64), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output_array, &_stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #156 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Add_output_0_output, AI_STATIC,
  156, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 64), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_stage1_b3_fuse_mode_Add_output_0_output_array, &_stage1_b3_fuse_mode_Add_output_0_output_array_intq)

/* Tensor #157 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_1_output_0_output, AI_STATIC,
  157, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 64), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_stage1_b3_fuse_mode_Mul_1_output_0_output_array, &_stage1_b3_fuse_mode_Mul_1_output_0_output_array_intq)

/* Tensor #158 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_2_output_0_output, AI_STATIC,
  158, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 64), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_stage1_b3_fuse_mode_Mul_2_output_0_output_array, &_stage1_b3_fuse_mode_Mul_2_output_0_output_array_intq)

/* Tensor #159 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_3_output_0_output, AI_STATIC,
  159, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 64), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_stage1_b3_fuse_mode_Mul_3_output_0_output_array, &_stage1_b3_fuse_mode_Mul_3_output_0_output_array_intq)

/* Tensor #160 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_output_0_output, AI_STATIC,
  160, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 64), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_stage1_b3_fuse_mode_Mul_output_0_output_array, &_stage1_b3_fuse_mode_Mul_output_0_output_array_intq)

/* Tensor #161 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_output, AI_STATIC,
  161, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_output_array, &_stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_output_array_intq)

/* Tensor #162 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Sub_output_0_output, AI_STATIC,
  162, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_stage1_b3_fuse_mode_Sub_output_0_output_array, NULL)

/* Tensor #163 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output, AI_STATIC,
  163, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, &_stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #164 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output0, AI_STATIC,
  164, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, &_stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #165 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_output, AI_STATIC,
  165, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_output_array, &_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_output_array_intq)

/* Tensor #166 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_scratch0, AI_STATIC,
  166, 0x0,
  AI_SHAPE_INIT(4, 1, 22, 1, 1), AI_STRIDE_INIT(4, 1, 1, 22, 22),
  1, &_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #167 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_weights, AI_STATIC,
  167, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 3, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_weights_array, &_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_weights_array_intq)

/* Tensor #168 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_output, AI_STATIC,
  168, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_output_array, NULL)

/* Tensor #169 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_output0, AI_STATIC,
  169, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_output_array, NULL)

/* Tensor #170 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output, AI_STATIC,
  170, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array_intq)

/* Tensor #171 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output0, AI_STATIC,
  171, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array_intq)

/* Tensor #172 */
AI_TENSOR_OBJ_DECLARE(
  _stage1_b3_shortcut_shortcut_0_AveragePool_output_0_output, AI_STATIC,
  172, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 9, 64), AI_STRIDE_INIT(4, 1, 1, 32, 288),
  1, &_stage1_b3_shortcut_shortcut_0_AveragePool_output_0_output_array, &_stage1_b3_shortcut_shortcut_0_AveragePool_output_0_output_array_intq)

/* Tensor #173 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_after_block_activation_Relu_output_0_0_conversion_0_0__stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output, AI_STATIC,
  173, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 4, 4, 224, 2016),
  1, &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_0_0__stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array, NULL)

/* Tensor #174 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_after_block_activation_Relu_output_0_0_conversion_output, AI_STATIC,
  174, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_output_array, &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_output_array_intq)

/* Tensor #175 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_output, AI_STATIC,
  175, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 9, 64), AI_STRIDE_INIT(4, 1, 1, 64, 576),
  1, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #176 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_output0, AI_STATIC,
  176, 0x1,
  AI_SHAPE_INIT(5, 1, 16, 9, 64, 4), AI_STRIDE_INIT(5, 1, 1, 64, 576, 16),
  1, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #177 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias, AI_STATIC,
  177, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array, NULL)

/* Tensor #178 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output, AI_STATIC,
  178, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 9, 64), AI_STRIDE_INIT(4, 4, 4, 256, 2304),
  1, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array, NULL)

/* Tensor #179 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights, AI_STATIC,
  179, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 64), AI_STRIDE_INIT(4, 4, 32, 2048, 2048),
  1, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array, NULL)

/* Tensor #180 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_output, AI_STATIC,
  180, 0x1,
  AI_SHAPE_INIT(5, 1, 4, 9, 64, 16), AI_STRIDE_INIT(5, 1, 1, 64, 576, 4),
  1, &_stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_output_array, &_stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #181 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_output0, AI_STATIC,
  181, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 9, 64), AI_STRIDE_INIT(4, 1, 1, 64, 576),
  1, &_stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_output_array, &_stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #182 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_1_act_Mul_output_0_0_0__stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output, AI_STATIC,
  182, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 9, 32), AI_STRIDE_INIT(4, 4, 4, 256, 2304),
  1, &_stage2_b4_block_block_1_act_Mul_output_0_0_0__stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array, NULL)

/* Tensor #183 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_1_act_Mul_output_0_output, AI_STATIC,
  183, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 9, 32), AI_STRIDE_INIT(4, 1, 1, 64, 576),
  1, &_stage2_b4_block_block_1_act_Mul_output_0_output_array, &_stage2_b4_block_block_1_act_Mul_output_0_output_array_intq)

/* Tensor #184 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_1_act_Sigmoid_output_0_output, AI_STATIC,
  184, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 9, 32), AI_STRIDE_INIT(4, 1, 1, 64, 576),
  1, &_stage2_b4_block_block_1_act_Sigmoid_output_0_output_array, &_stage2_b4_block_block_1_act_Sigmoid_output_0_output_array_intq)

/* Tensor #185 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_freq_Conv_output_0_output, AI_STATIC,
  185, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 9, 32), AI_STRIDE_INIT(4, 1, 1, 64, 576),
  1, &_stage2_b4_block_block_1_dw_freq_Conv_output_0_output_array, &_stage2_b4_block_block_1_dw_freq_Conv_output_0_output_array_intq)

/* Tensor #186 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_freq_Conv_output_0_scratch0, AI_STATIC,
  186, 0x0,
  AI_SHAPE_INIT(4, 1, 1601, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1601, 1601),
  1, &_stage2_b4_block_block_1_dw_freq_Conv_output_0_scratch0_array, NULL)

/* Tensor #187 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_freq_Conv_output_0_weights, AI_STATIC,
  187, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 5, 1), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &_stage2_b4_block_block_1_dw_freq_Conv_output_0_weights_array, &_stage2_b4_block_block_1_dw_freq_Conv_output_0_weights_array_intq)

/* Tensor #188 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_time_Conv_output_0_bias, AI_STATIC,
  188, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_stage2_b4_block_block_1_dw_time_Conv_output_0_bias_array, NULL)

/* Tensor #189 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_time_Conv_output_0_output, AI_STATIC,
  189, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 9, 32), AI_STRIDE_INIT(4, 1, 1, 64, 576),
  1, &_stage2_b4_block_block_1_dw_time_Conv_output_0_output_array, &_stage2_b4_block_block_1_dw_time_Conv_output_0_output_array_intq)

/* Tensor #190 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_time_Conv_output_0_scratch0, AI_STATIC,
  190, 0x0,
  AI_SHAPE_INIT(4, 1, 1217, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1217, 1217),
  1, &_stage2_b4_block_block_1_dw_time_Conv_output_0_scratch0_array, NULL)

/* Tensor #191 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_time_Conv_output_0_weights, AI_STATIC,
  191, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 1, 1), AI_STRIDE_INIT(4, 1, 64, 64, 192),
  1, &_stage2_b4_block_block_1_dw_time_Conv_output_0_weights_array, &_stage2_b4_block_block_1_dw_time_Conv_output_0_weights_array_intq)

/* Tensor #192 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_output, AI_STATIC,
  192, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #193 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_output0, AI_STATIC,
  193, 0x1,
  AI_SHAPE_INIT(5, 1, 14, 9, 32, 4), AI_STRIDE_INIT(5, 1, 1, 56, 504, 14),
  1, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #194 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias, AI_STATIC,
  194, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array, NULL)

/* Tensor #195 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output, AI_STATIC,
  195, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 4, 4, 224, 2016),
  1, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array, NULL)

/* Tensor #196 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights, AI_STATIC,
  196, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 56), AI_STRIDE_INIT(4, 4, 64, 3584, 3584),
  1, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array, NULL)

/* Tensor #197 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_output, AI_STATIC,
  197, 0x1,
  AI_SHAPE_INIT(5, 1, 4, 9, 32, 14), AI_STRIDE_INIT(5, 1, 1, 56, 504, 4),
  1, &_stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_output_array, &_stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #198 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_output0, AI_STATIC,
  198, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_output_array, &_stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #199 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_after_block_activation_Relu_output_0_output, AI_STATIC,
  199, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_stage2_b5_after_block_activation_Relu_output_0_output_array, &_stage2_b5_after_block_activation_Relu_output_0_output_array_intq)

/* Tensor #200 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_output, AI_STATIC,
  200, 0x1,
  AI_SHAPE_INIT(4, 1, 120, 9, 32), AI_STRIDE_INIT(4, 1, 1, 120, 1080),
  1, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #201 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_output0, AI_STATIC,
  201, 0x1,
  AI_SHAPE_INIT(5, 1, 30, 9, 32, 4), AI_STRIDE_INIT(5, 1, 1, 120, 1080, 30),
  1, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #202 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias, AI_STATIC,
  202, 0x0,
  AI_SHAPE_INIT(4, 1, 120, 1, 1), AI_STRIDE_INIT(4, 4, 4, 480, 480),
  1, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array, NULL)

/* Tensor #203 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output, AI_STATIC,
  203, 0x0,
  AI_SHAPE_INIT(4, 1, 120, 9, 32), AI_STRIDE_INIT(4, 4, 4, 480, 4320),
  1, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array, NULL)

/* Tensor #204 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights, AI_STATIC,
  204, 0x0,
  AI_SHAPE_INIT(4, 14, 1, 1, 120), AI_STRIDE_INIT(4, 4, 56, 6720, 6720),
  1, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array, NULL)

/* Tensor #205 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_output, AI_STATIC,
  205, 0x1,
  AI_SHAPE_INIT(5, 1, 4, 9, 32, 30), AI_STRIDE_INIT(5, 1, 1, 120, 1080, 4),
  1, &_stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_output_array, &_stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #206 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_output0, AI_STATIC,
  206, 0x1,
  AI_SHAPE_INIT(4, 1, 120, 9, 32), AI_STRIDE_INIT(4, 1, 1, 120, 1080),
  1, &_stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_output_array, &_stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #207 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_1_act_Mul_output_0_0_0__stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output, AI_STATIC,
  207, 0x0,
  AI_SHAPE_INIT(4, 1, 120, 9, 32), AI_STRIDE_INIT(4, 4, 4, 480, 4320),
  1, &_stage2_b5_block_block_1_act_Mul_output_0_0_0__stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array, NULL)

/* Tensor #208 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_1_act_Mul_output_0_output, AI_STATIC,
  208, 0x1,
  AI_SHAPE_INIT(4, 1, 120, 9, 32), AI_STRIDE_INIT(4, 1, 1, 120, 1080),
  1, &_stage2_b5_block_block_1_act_Mul_output_0_output_array, &_stage2_b5_block_block_1_act_Mul_output_0_output_array_intq)

/* Tensor #209 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_1_act_Sigmoid_output_0_output, AI_STATIC,
  209, 0x1,
  AI_SHAPE_INIT(4, 1, 120, 9, 32), AI_STRIDE_INIT(4, 1, 1, 120, 1080),
  1, &_stage2_b5_block_block_1_act_Sigmoid_output_0_output_array, &_stage2_b5_block_block_1_act_Sigmoid_output_0_output_array_intq)

/* Tensor #210 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_freq_Conv_output_0_bias, AI_STATIC,
  210, 0x0,
  AI_SHAPE_INIT(4, 1, 120, 1, 1), AI_STRIDE_INIT(4, 4, 4, 480, 480),
  1, &_stage2_b5_block_block_1_dw_freq_Conv_output_0_bias_array, NULL)

/* Tensor #211 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_freq_Conv_output_0_output, AI_STATIC,
  211, 0x1,
  AI_SHAPE_INIT(4, 1, 120, 9, 32), AI_STRIDE_INIT(4, 1, 1, 120, 1080),
  1, &_stage2_b5_block_block_1_dw_freq_Conv_output_0_output_array, &_stage2_b5_block_block_1_dw_freq_Conv_output_0_output_array_intq)

/* Tensor #212 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_freq_Conv_output_0_scratch0, AI_STATIC,
  212, 0x0,
  AI_SHAPE_INIT(4, 1, 3001, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3001, 3001),
  1, &_stage2_b5_block_block_1_dw_freq_Conv_output_0_scratch0_array, NULL)

/* Tensor #213 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_freq_Conv_output_0_weights, AI_STATIC,
  213, 0x1,
  AI_SHAPE_INIT(4, 120, 1, 5, 1), AI_STRIDE_INIT(4, 1, 120, 120, 120),
  1, &_stage2_b5_block_block_1_dw_freq_Conv_output_0_weights_array, &_stage2_b5_block_block_1_dw_freq_Conv_output_0_weights_array_intq)

/* Tensor #214 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_time_Conv_output_0_bias, AI_STATIC,
  214, 0x0,
  AI_SHAPE_INIT(4, 1, 120, 1, 1), AI_STRIDE_INIT(4, 4, 4, 480, 480),
  1, &_stage2_b5_block_block_1_dw_time_Conv_output_0_bias_array, NULL)

/* Tensor #215 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_time_Conv_output_0_output, AI_STATIC,
  215, 0x1,
  AI_SHAPE_INIT(4, 1, 120, 9, 32), AI_STRIDE_INIT(4, 1, 1, 120, 1080),
  1, &_stage2_b5_block_block_1_dw_time_Conv_output_0_output_array, &_stage2_b5_block_block_1_dw_time_Conv_output_0_output_array_intq)

/* Tensor #216 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_time_Conv_output_0_scratch0, AI_STATIC,
  216, 0x0,
  AI_SHAPE_INIT(4, 1, 2281, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2281, 2281),
  1, &_stage2_b5_block_block_1_dw_time_Conv_output_0_scratch0_array, NULL)

/* Tensor #217 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_time_Conv_output_0_weights, AI_STATIC,
  217, 0x1,
  AI_SHAPE_INIT(4, 120, 3, 1, 1), AI_STRIDE_INIT(4, 1, 120, 120, 360),
  1, &_stage2_b5_block_block_1_dw_time_Conv_output_0_weights_array, &_stage2_b5_block_block_1_dw_time_Conv_output_0_weights_array_intq)

/* Tensor #218 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_output, AI_STATIC,
  218, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #219 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_output0, AI_STATIC,
  219, 0x1,
  AI_SHAPE_INIT(5, 1, 14, 9, 32, 4), AI_STRIDE_INIT(5, 1, 1, 56, 504, 14),
  1, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array_intq)

/* Tensor #220 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias, AI_STATIC,
  220, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array, NULL)

/* Tensor #221 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output, AI_STATIC,
  221, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 4, 4, 224, 2016),
  1, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array, NULL)

/* Tensor #222 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights, AI_STATIC,
  222, 0x0,
  AI_SHAPE_INIT(4, 30, 1, 1, 56), AI_STRIDE_INIT(4, 4, 120, 6720, 6720),
  1, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array, NULL)

/* Tensor #223 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output, AI_STATIC,
  223, 0x1,
  AI_SHAPE_INIT(5, 1, 4, 9, 32, 14), AI_STRIDE_INIT(5, 1, 1, 56, 504, 4),
  1, &_stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output_array, &_stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #224 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output0, AI_STATIC,
  224, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output_array, &_stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output_array_intq)

/* Tensor #225 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Add_output_0_output, AI_STATIC,
  225, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_stage2_b5_fuse_mode_Add_output_0_output_array, &_stage2_b5_fuse_mode_Add_output_0_output_array_intq)

/* Tensor #226 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_1_output_0_output, AI_STATIC,
  226, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_stage2_b5_fuse_mode_Mul_1_output_0_output_array, &_stage2_b5_fuse_mode_Mul_1_output_0_output_array_intq)

/* Tensor #227 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_2_output_0_output, AI_STATIC,
  227, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_stage2_b5_fuse_mode_Mul_2_output_0_output_array, &_stage2_b5_fuse_mode_Mul_2_output_0_output_array_intq)

/* Tensor #228 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_3_output_0_output, AI_STATIC,
  228, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_stage2_b5_fuse_mode_Mul_3_output_0_output_array, &_stage2_b5_fuse_mode_Mul_3_output_0_output_array_intq)

/* Tensor #229 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_output_0_output, AI_STATIC,
  229, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 9, 32), AI_STRIDE_INIT(4, 1, 1, 56, 504),
  1, &_stage2_b5_fuse_mode_Mul_output_0_output_array, &_stage2_b5_fuse_mode_Mul_output_0_output_array_intq)

/* Tensor #230 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_output, AI_STATIC,
  230, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 1, 1, 56, 56),
  1, &_stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_output_array, &_stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_output_array_intq)

/* Tensor #231 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Sub_output_0_output, AI_STATIC,
  231, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &_stage2_b5_fuse_mode_Sub_output_0_output_array, NULL)

/* Tensor #232 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output, AI_STATIC,
  232, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 1, 1, 56, 56),
  1, &_stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, &_stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #233 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output0, AI_STATIC,
  233, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 56), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array, &_stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #234 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_output, AI_STATIC,
  234, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 56), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_output_array, &_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_output_array_intq)

/* Tensor #235 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_scratch0, AI_STATIC,
  235, 0x0,
  AI_SHAPE_INIT(4, 1, 22, 1, 1), AI_STRIDE_INIT(4, 1, 1, 22, 22),
  1, &_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #236 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_weights, AI_STATIC,
  236, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 3, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_weights_array, &_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_weights_array_intq)

/* Tensor #237 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_output, AI_STATIC,
  237, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 56), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_output_array, NULL)

/* Tensor #238 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_output0, AI_STATIC,
  238, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_output_array, NULL)

/* Tensor #239 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output, AI_STATIC,
  239, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 56), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array_intq)

/* Tensor #240 */
AI_TENSOR_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output0, AI_STATIC,
  240, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 1, 1, 56, 56),
  1, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array_intq)

/* Tensor #241 */
AI_TENSOR_OBJ_DECLARE(
  logmel_output, AI_STATIC,
  241, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 65, 256), AI_STRIDE_INIT(4, 1, 1, 1, 65),
  1, &logmel_output_array, &logmel_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  _detection_head_detection_head_2_GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_detection_head_detection_head_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_detection_head_detection_head_2_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _detection_head_detection_head_2_GlobalAveragePool_output_0_layer, 473,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_detection_head_detection_head_2_GlobalAveragePool_output_0_chain,
  NULL, &_detection_head_detection_head_2_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(9, 32), 
  .pool_stride = AI_SHAPE_2D_INIT(9, 32), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _detection_head_detection_head_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_detection_head_detection_head_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_detection_head_detection_head_0_Conv_output_0_weights, &_detection_head_detection_head_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_detection_head_detection_head_0_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _detection_head_detection_head_0_Conv_output_0_layer, 470,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_pw_sssa8_ch,
  &_detection_head_detection_head_0_Conv_output_0_chain,
  NULL, &_detection_head_detection_head_2_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _aff_fusion_Add_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_aff_fusion_Mul_1_output_0_output, &_aff_fusion_Mul_3_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _aff_fusion_Add_1_output_0_layer, 467,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_aff_fusion_Add_1_output_0_chain,
  NULL, &_detection_head_detection_head_0_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _aff_fusion_Mul_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_aff_fusion_Mul_2_output_0_output, &_aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_Mul_3_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _aff_fusion_Mul_3_output_0_layer, 464,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_aff_fusion_Mul_3_output_0_chain,
  NULL, &_aff_fusion_Add_1_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _aff_fusion_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_channel_adjust_Conv_output_0_output, &_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _aff_fusion_Mul_2_output_0_layer, 327,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_aff_fusion_Mul_2_output_0_chain,
  NULL, &_aff_fusion_Mul_3_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _aff_fusion_Mul_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_aff_fusion_Mul_output_0_output, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_Mul_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _aff_fusion_Mul_1_output_0_layer, 458,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_aff_fusion_Mul_1_output_0_chain,
  NULL, &_aff_fusion_Mul_2_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_layer, 459,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_chain,
  NULL, &_aff_fusion_Mul_1_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _aff_fusion_Sub_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b1_fuse_mode_Constant_2_output_0_4D, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _aff_fusion_Sub_output_0_layer, 459,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_aff_fusion_Sub_output_0_chain,
  NULL, &_aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_layer, 449,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_chain,
  NULL, &_aff_fusion_Sub_output_0_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -116, -115, -113, -111, -109, -106, -104, -100, -97, -93, -89, -85, -80, -74, -68, -62, -56, -48, -41, -33, -25, -17, -9, 0, 8, 16, 24, 32, 40, 47, 55, 61, 67, 73, 79, 84, 88, 92, 96, 99, 103, 105, 108, 110, 112, 114, 115, 117, 118, 119, 120, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_nl_params_data, _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_channel_att_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_layer, 449,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_chain,
  NULL, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_layer, AI_STATIC, 
  .nl_params = &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _aff_fusion_channel_att_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_channel_att_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_aff_fusion_channel_att_conv_Conv_output_0_weights, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_channel_att_conv_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _aff_fusion_channel_att_conv_Conv_output_0_layer, 446,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &_aff_fusion_channel_att_conv_Conv_output_0_chain,
  NULL, &_aff_fusion_channel_att_sigmoid_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_layer, 437,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_chain,
  NULL, &_aff_fusion_channel_att_conv_Conv_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(9, 32), 
  .pool_stride = AI_SHAPE_2D_INIT(9, 32), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _aff_fusion_Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage2_b5_after_block_activation_Relu_output_0_output, &_channel_adjust_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _aff_fusion_Add_output_0_layer, 431,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_aff_fusion_Add_output_0_chain,
  NULL, &_aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _channel_adjust_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_channel_adjust_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_channel_adjust_Conv_output_0_weights, &_channel_adjust_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_channel_adjust_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _channel_adjust_Conv_output_0_layer, 321,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_pw_sssa8_ch,
  &_channel_adjust_Conv_output_0_chain,
  NULL, &_aff_fusion_Add_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Mul_output_0_output, &_Mul_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_output_0_layer, 315,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Add_output_0_chain,
  NULL, &_channel_adjust_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Sigmoid_output_0_DequantizeLinear_Output_const_4D, &_AveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Mul_output_0_layer, 306,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Mul_output_0_chain,
  NULL, &_Add_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _AveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_after_block_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_AveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _AveragePool_output_0_layer, 297,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_AveragePool_output_0_chain,
  NULL, &_Mul_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .count_include_pad = 0, 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Mul_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Sub_output_0_DequantizeLinear_Output_const_4D, &_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Mul_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Mul_1_output_0_layer, 307,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Mul_1_output_0_chain,
  NULL, &_AveragePool_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_after_block_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _MaxPool_output_0_layer, 298,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &_MaxPool_output_0_chain,
  NULL, &_Mul_1_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _aff_fusion_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage2_b5_after_block_activation_Relu_output_0_output, &_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_aff_fusion_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _aff_fusion_Mul_output_0_layer, 432,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_aff_fusion_Mul_output_0_chain,
  NULL, &_MaxPool_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_after_block_activation_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage2_b5_fuse_mode_Mul_1_output_0_output, &_stage2_b5_fuse_mode_Mul_3_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_after_block_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_after_block_activation_Relu_output_0_layer, 428,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage2_b5_after_block_activation_Relu_output_0_chain,
  NULL, &_aff_fusion_Mul_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage2_b5_fuse_mode_Mul_2_output_0_output, &_stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_Mul_3_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_3_output_0_layer, 425,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage2_b5_fuse_mode_Mul_3_output_0_chain,
  NULL, &_stage2_b5_after_block_activation_Relu_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_output, &_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_2_output_0_layer, 354,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage2_b5_fuse_mode_Mul_2_output_0_chain,
  NULL, &_stage2_b5_fuse_mode_Mul_3_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage2_b5_fuse_mode_Mul_output_0_output, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_Mul_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_1_output_0_layer, 419,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage2_b5_fuse_mode_Mul_1_output_0_chain,
  NULL, &_stage2_b5_fuse_mode_Mul_2_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_layer, 420,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_chain,
  NULL, &_stage2_b5_fuse_mode_Mul_1_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Sub_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b1_fuse_mode_Constant_2_output_0_4D, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Sub_output_0_layer, 420,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_stage2_b5_fuse_mode_Sub_output_0_chain,
  NULL, &_stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_layer, 410,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_chain,
  NULL, &_stage2_b5_fuse_mode_Sub_output_0_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params_data[] = { -127, -127, -126, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -122, -121, -121, -120, -120, -119, -118, -118, -117, -116, -116, -115, -114, -113, -112, -111, -110, -108, -107, -106, -104, -103, -101, -100, -98, -96, -94, -92, -90, -87, -85, -82, -80, -77, -74, -71, -68, -65, -61, -58, -54, -51, -47, -43, -39, -35, -31, -27, -22, -18, -14, -9, -5, 0, 4, 8, 13, 17, 21, 26, 30, 34, 38, 42, 46, 50, 53, 57, 60, 64, 67, 70, 73, 76, 79, 81, 84, 86, 89, 91, 93, 95, 97, 99, 100, 102, 103, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 115, 116, 117, 117, 118, 119, 119, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params_data, _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_layer, 410,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_chain,
  NULL, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_layer, AI_STATIC, 
  .nl_params = &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_weights, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_layer, 407,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_chain,
  NULL, &_stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_layer, 398,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_chain,
  NULL, &_stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(9, 32), 
  .pool_stride = AI_SHAPE_2D_INIT(9, 32), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output0, &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Add_output_0_layer, 392,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage2_b5_fuse_mode_Add_output_0_chain,
  NULL, &_stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output0, &_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_fuse_mode_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_fuse_mode_Mul_output_0_layer, 393,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage2_b5_fuse_mode_Mul_output_0_chain,
  NULL, &_stage2_b5_fuse_mode_Add_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_layer, 386,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_chain,
  NULL, &_stage2_b5_fuse_mode_Mul_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_layer, 380,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_chain,
  NULL, &_stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_1_act_Mul_output_0_0_0__stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_layer, 380,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_chain,
  NULL, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_layer, AI_STATIC, 
  .groups = 4, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_block_block_1_act_Mul_output_0_0_0__stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_1_act_Mul_output_0_0_0__stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_block_block_1_act_Mul_output_0_0_0__stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_layer, 377,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage2_b5_block_block_1_act_Mul_output_0_0_0__stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_chain,
  NULL, &_stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_block_block_1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage2_b5_block_block_1_dw_time_Conv_output_0_output, &_stage2_b5_block_block_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_block_block_1_act_Mul_output_0_layer, 377,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage2_b5_block_block_1_act_Mul_output_0_chain,
  NULL, &_stage2_b5_block_block_1_act_Mul_output_0_0_0__stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 _stage2_b5_block_block_1_act_Sigmoid_output_0_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -115, -113, -111, -109, -107, -104, -101, -97, -94, -90, -85, -80, -75, -69, -63, -56, -49, -41, -34, -26, -17, -9, 0, 8, 16, 25, 33, 40, 48, 55, 62, 68, 74, 79, 84, 89, 93, 96, 100, 103, 106, 108, 110, 112, 114, 116, 117, 118, 119, 120, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    _stage2_b5_block_block_1_act_Sigmoid_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _stage2_b5_block_block_1_act_Sigmoid_output_0_nl_params_data, _stage2_b5_block_block_1_act_Sigmoid_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_block_block_1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_1_dw_time_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_block_block_1_act_Sigmoid_output_0_layer, 374,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_stage2_b5_block_block_1_act_Sigmoid_output_0_chain,
  NULL, &_stage2_b5_block_block_1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = &_stage2_b5_block_block_1_act_Sigmoid_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_time_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_1_dw_freq_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_1_dw_time_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage2_b5_block_block_1_dw_time_Conv_output_0_weights, &_stage2_b5_block_block_1_dw_time_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_1_dw_time_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_time_Conv_output_0_layer, 371,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_sssa8_ch,
  &_stage2_b5_block_block_1_dw_time_Conv_output_0_chain,
  NULL, &_stage2_b5_block_block_1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 120, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_freq_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_1_dw_freq_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage2_b5_block_block_1_dw_freq_Conv_output_0_weights, &_stage2_b5_block_block_1_dw_freq_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_1_dw_freq_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_block_block_1_dw_freq_Conv_output_0_layer, 368,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_sssa8_ch,
  &_stage2_b5_block_block_1_dw_freq_Conv_output_0_chain,
  NULL, &_stage2_b5_block_block_1_dw_time_Conv_output_0_layer, AI_STATIC, 
  .groups = 120, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 2, 0, 2, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_layer, 362,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_chain,
  NULL, &_stage2_b5_block_block_1_dw_freq_Conv_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_layer, 353,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_chain,
  NULL, &_stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_0_0__stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_layer, 353,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_chain,
  NULL, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_layer, AI_STATIC, 
  .groups = 4, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_after_block_activation_Relu_output_0_0_conversion_0_0__stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_0_0__stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_after_block_activation_Relu_output_0_0_conversion_0_0__stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_layer, 350,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_0_0__stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_chain,
  NULL, &_stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_after_block_activation_Relu_output_0_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_after_block_activation_Relu_output_0_0_conversion_layer, 350,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_chain,
  NULL, &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_0_0__stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_layer, 347,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_chain,
  NULL, &_stage2_b4_after_block_activation_Relu_output_0_0_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_layer, 341,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_chain,
  NULL, &_stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_1_act_Mul_output_0_0_0__stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_layer, 341,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_chain,
  NULL, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_layer, AI_STATIC, 
  .groups = 4, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_block_block_1_act_Mul_output_0_0_0__stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_1_act_Mul_output_0_0_0__stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_block_block_1_act_Mul_output_0_0_0__stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_layer, 338,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage2_b4_block_block_1_act_Mul_output_0_0_0__stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_chain,
  NULL, &_stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_block_block_1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage2_b4_block_block_1_dw_time_Conv_output_0_output, &_stage2_b4_block_block_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_block_block_1_act_Mul_output_0_layer, 338,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage2_b4_block_block_1_act_Mul_output_0_chain,
  NULL, &_stage2_b4_block_block_1_act_Mul_output_0_0_0__stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 _stage2_b4_block_block_1_act_Sigmoid_output_0_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -120, -119, -118, -117, -116, -115, -113, -111, -110, -108, -105, -103, -100, -98, -94, -91, -87, -84, -79, -75, -70, -65, -59, -54, -48, -41, -35, -28, -22, -15, -8, 0, 7, 14, 21, 27, 34, 40, 47, 53, 58, 64, 69, 74, 78, 83, 86, 90, 93, 97, 99, 102, 104, 107, 109, 110, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    _stage2_b4_block_block_1_act_Sigmoid_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _stage2_b4_block_block_1_act_Sigmoid_output_0_nl_params_data, _stage2_b4_block_block_1_act_Sigmoid_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_block_block_1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_1_dw_time_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_block_block_1_act_Sigmoid_output_0_layer, 335,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_stage2_b4_block_block_1_act_Sigmoid_output_0_chain,
  NULL, &_stage2_b4_block_block_1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = &_stage2_b4_block_block_1_act_Sigmoid_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_time_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_1_dw_freq_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_1_dw_time_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage2_b4_block_block_1_dw_time_Conv_output_0_weights, &_stage2_b4_block_block_1_dw_time_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_1_dw_time_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_time_Conv_output_0_layer, 332,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_sssa8_ch,
  &_stage2_b4_block_block_1_dw_time_Conv_output_0_chain,
  NULL, &_stage2_b4_block_block_1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_freq_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_1_dw_freq_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage2_b4_block_block_1_dw_freq_Conv_output_0_weights, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_1_dw_freq_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_block_block_1_dw_freq_Conv_output_0_layer, 326,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_sssa8_ch,
  &_stage2_b4_block_block_1_dw_freq_Conv_output_0_chain,
  NULL, &_stage2_b4_block_block_1_dw_time_Conv_output_0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 2, 0, 2, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_layer, 314,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_chain,
  NULL, &_stage2_b4_block_block_1_dw_freq_Conv_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_layer, 296,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_chain,
  NULL, &_stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_after_block_activation_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_layer, 296,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_chain,
  NULL, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_layer, AI_STATIC, 
  .groups = 4, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_after_block_activation_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_after_block_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_after_block_activation_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_after_block_activation_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_layer, 293,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b3_after_block_activation_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_chain,
  NULL, &_stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_after_block_activation_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b3_fuse_mode_Mul_1_output_0_output, &_stage1_b3_fuse_mode_Mul_3_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_after_block_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_after_block_activation_Relu_output_0_layer, 293,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b3_after_block_activation_Relu_output_0_chain,
  NULL, &_stage1_b3_after_block_activation_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b3_fuse_mode_Mul_2_output_0_output, &_stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_Mul_3_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_3_output_0_layer, 290,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b3_fuse_mode_Mul_3_output_0_chain,
  NULL, &_stage1_b3_after_block_activation_Relu_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b3_shortcut_shortcut_0_AveragePool_output_0_output, &_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_2_output_0_layer, 222,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b3_fuse_mode_Mul_2_output_0_chain,
  NULL, &_stage1_b3_fuse_mode_Mul_3_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b3_fuse_mode_Mul_output_0_output, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_Mul_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_1_output_0_layer, 284,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b3_fuse_mode_Mul_1_output_0_chain,
  NULL, &_stage1_b3_fuse_mode_Mul_2_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_layer, 285,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_chain,
  NULL, &_stage1_b3_fuse_mode_Mul_1_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Sub_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b1_fuse_mode_Constant_2_output_0_4D, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Sub_output_0_layer, 285,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_stage1_b3_fuse_mode_Sub_output_0_chain,
  NULL, &_stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_layer, 275,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_chain,
  NULL, &_stage1_b3_fuse_mode_Sub_output_0_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params_data[] = { -65, -64, -64, -63, -63, -62, -62, -61, -61, -60, -60, -59, -59, -58, -58, -57, -57, -56, -56, -55, -55, -54, -54, -53, -52, -52, -51, -51, -50, -50, -49, -49, -48, -47, -47, -46, -46, -45, -45, -44, -43, -43, -42, -42, -41, -40, -40, -39, -38, -38, -37, -37, -36, -35, -35, -34, -33, -33, -32, -32, -31, -30, -30, -29, -28, -28, -27, -26, -26, -25, -24, -24, -23, -22, -21, -21, -20, -19, -19, -18, -17, -17, -16, -15, -14, -14, -13, -12, -11, -11, -10, -9, -9, -8, -7, -6, -6, -5, -4, -3, -3, -2, -1, 0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 25, 26, 27, 28, 29, 30, 30, 31, 32, 33, 34, 35, 36, 36, 37, 38, 39, 40, 41, 41, 42, 43, 44, 45, 46, 47, 47, 48, 49, 50, 51, 52, 53, 54, 54, 55, 56, 57, 58, 59, 60, 60, 61, 62, 63, 64, 65, 66, 67, 67, 68, 69, 70, 71, 72, 73, 74, 74, 75, 76, 77, 78, 79, 80, 80, 81, 82, 83, 84, 85, 86, 87, 87, 88, 89, 90, 91, 92, 93, 93, 94, 95, 96, 97, 98, 99, 99, 100, 101, 102, 103, 104, 104, 105, 106, 107, 108, 109, 109, 110, 111, 112, 113, 114, 114, 115, 116, 117, 118, 118, 119, 120, 121, 122, 122, 123, 124, 125, 126, 126, 127 };
AI_ARRAY_OBJ_DECLARE(
    _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params_data, _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_layer, 275,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_chain,
  NULL, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_layer, AI_STATIC, 
  .nl_params = &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_weights, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_layer, 272,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_chain,
  NULL, &_stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_layer, 263,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_chain,
  NULL, &_stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(9, 64), 
  .pool_stride = AI_SHAPE_2D_INIT(9, 64), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output0, &_stage1_b3_shortcut_shortcut_0_AveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Add_output_0_layer, 257,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b3_fuse_mode_Add_output_0_chain,
  NULL, &_stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_shortcut_shortcut_0_AveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_after_block_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_shortcut_shortcut_0_AveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_shortcut_shortcut_0_AveragePool_output_0_layer, 216,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_stage1_b3_shortcut_shortcut_0_AveragePool_output_0_chain,
  NULL, &_stage1_b3_fuse_mode_Add_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(3, 3), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 1), 
  .count_include_pad = 1, 
  .pool_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output0, &_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_fuse_mode_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_fuse_mode_Mul_output_0_layer, 258,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b3_fuse_mode_Mul_output_0_chain,
  NULL, &_stage1_b3_shortcut_shortcut_0_AveragePool_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_layer, 251,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_chain,
  NULL, &_stage1_b3_fuse_mode_Mul_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_layer, 245,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_chain,
  NULL, &_stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_1_act_Mul_output_0_0_0__stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_layer, 245,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_chain,
  NULL, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_layer, AI_STATIC, 
  .groups = 4, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_block_block_1_act_Mul_output_0_0_0__stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_1_act_Mul_output_0_0_0__stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_block_block_1_act_Mul_output_0_0_0__stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_layer, 242,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b3_block_block_1_act_Mul_output_0_0_0__stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_chain,
  NULL, &_stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_block_block_1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b3_block_block_1_dw_time_Conv_output_0_output, &_stage1_b3_block_block_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_block_block_1_act_Mul_output_0_layer, 242,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b3_block_block_1_act_Mul_output_0_chain,
  NULL, &_stage1_b3_block_block_1_act_Mul_output_0_0_0__stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 _stage1_b3_block_block_1_act_Sigmoid_output_0_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -125, -125, -125, -124, -123, -123, -122, -121, -120, -119, -117, -116, -114, -111, -109, -106, -103, -100, -96, -91, -86, -81, -75, -68, -61, -54, -46, -37, -28, -19, -10, -1, 9, 18, 27, 36, 45, 53, 60, 67, 74, 80, 85, 90, 95, 99, 102, 105, 108, 110, 113, 115, 116, 118, 119, 120, 121, 122, 122, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    _stage1_b3_block_block_1_act_Sigmoid_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _stage1_b3_block_block_1_act_Sigmoid_output_0_nl_params_data, _stage1_b3_block_block_1_act_Sigmoid_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_block_block_1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_1_dw_time_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_block_block_1_act_Sigmoid_output_0_layer, 239,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_stage1_b3_block_block_1_act_Sigmoid_output_0_chain,
  NULL, &_stage1_b3_block_block_1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = &_stage1_b3_block_block_1_act_Sigmoid_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_time_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_1_dw_freq_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_1_dw_time_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b3_block_block_1_dw_time_Conv_output_0_weights, &_stage1_b3_block_block_1_dw_time_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_1_dw_time_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_time_Conv_output_0_layer, 236,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_sssa8_ch,
  &_stage1_b3_block_block_1_dw_time_Conv_output_0_chain,
  NULL, &_stage1_b3_block_block_1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_freq_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_1_dw_freq_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b3_block_block_1_dw_freq_Conv_output_0_weights, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_1_dw_freq_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_block_block_1_dw_freq_Conv_output_0_layer, 233,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_sssa8_ch,
  &_stage1_b3_block_block_1_dw_freq_Conv_output_0_chain,
  NULL, &_stage1_b3_block_block_1_dw_time_Conv_output_0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_layer, 227,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_chain,
  NULL, &_stage1_b3_block_block_1_dw_freq_Conv_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_layer, 215,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_chain,
  NULL, &_stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_after_block_activation_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_layer, 215,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_chain,
  NULL, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_layer, AI_STATIC, 
  .groups = 4, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_after_block_activation_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_after_block_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_after_block_activation_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_after_block_activation_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_layer, 212,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b2_after_block_activation_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_chain,
  NULL, &_stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_after_block_activation_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b2_fuse_mode_Mul_1_output_0_output, &_stage1_b2_fuse_mode_Mul_3_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_after_block_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_after_block_activation_Relu_output_0_layer, 212,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b2_after_block_activation_Relu_output_0_chain,
  NULL, &_stage1_b2_after_block_activation_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b2_fuse_mode_Mul_2_output_0_output, &_stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_Mul_3_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_3_output_0_layer, 209,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b2_fuse_mode_Mul_3_output_0_chain,
  NULL, &_stage1_b2_after_block_activation_Relu_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b1_after_block_activation_Relu_output_0_output, &_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_2_output_0_layer, 138,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b2_fuse_mode_Mul_2_output_0_chain,
  NULL, &_stage1_b2_fuse_mode_Mul_3_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b2_fuse_mode_Mul_output_0_output, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_Mul_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_1_output_0_layer, 203,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b2_fuse_mode_Mul_1_output_0_chain,
  NULL, &_stage1_b2_fuse_mode_Mul_2_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_layer, 204,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_chain,
  NULL, &_stage1_b2_fuse_mode_Mul_1_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Sub_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b1_fuse_mode_Constant_2_output_0_4D, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Sub_output_0_layer, 204,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_stage1_b2_fuse_mode_Sub_output_0_chain,
  NULL, &_stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_layer, 194,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_chain,
  NULL, &_stage1_b2_fuse_mode_Sub_output_0_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params_data[] = { -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 57, 58, 59, 60, 61, 62, 63, 63, 64, 65, 66, 67, 68, 68, 69, 70, 71, 71, 72, 73, 74, 75, 75, 76, 77, 77, 78, 79, 80, 80, 81, 82, 82, 83, 84, 84, 85, 85, 86, 87, 87, 88, 89, 89, 90, 90, 91, 91, 92, 93, 93, 94, 94, 95, 95, 96, 96, 97, 97, 98, 98, 99, 99, 100, 100, 101, 101, 102, 102, 102, 103, 103, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 113, 114, 114, 114, 115, 115, 115, 115, 116, 116, 116, 116, 117, 117, 117, 117, 117, 118, 118, 118, 118, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120, 121, 121, 121, 121, 121, 121, 122, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params_data, _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_layer, 194,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_chain,
  NULL, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_layer, AI_STATIC, 
  .nl_params = &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_weights, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_layer, 191,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_chain,
  NULL, &_stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_layer, 182,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_chain,
  NULL, &_stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(17, 64), 
  .pool_stride = AI_SHAPE_2D_INIT(17, 64), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output0, &_stage1_b1_after_block_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Add_output_0_layer, 176,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b2_fuse_mode_Add_output_0_chain,
  NULL, &_stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output0, &_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_fuse_mode_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_fuse_mode_Mul_output_0_layer, 177,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b2_fuse_mode_Mul_output_0_chain,
  NULL, &_stage1_b2_fuse_mode_Add_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_layer, 170,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_chain,
  NULL, &_stage1_b2_fuse_mode_Mul_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_layer, 164,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_chain,
  NULL, &_stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_1_act_Mul_output_0_0_0__stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_layer, 164,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_chain,
  NULL, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_layer, AI_STATIC, 
  .groups = 4, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_block_block_1_act_Mul_output_0_0_0__stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_1_act_Mul_output_0_0_0__stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_block_block_1_act_Mul_output_0_0_0__stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_layer, 161,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b2_block_block_1_act_Mul_output_0_0_0__stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_chain,
  NULL, &_stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_block_block_1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b2_block_block_1_dw_time_Conv_output_0_output, &_stage1_b2_block_block_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_block_block_1_act_Mul_output_0_layer, 161,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b2_block_block_1_act_Mul_output_0_chain,
  NULL, &_stage1_b2_block_block_1_act_Mul_output_0_0_0__stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 _stage1_b2_block_block_1_act_Sigmoid_output_0_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -125, -125, -124, -124, -123, -122, -121, -120, -118, -117, -115, -113, -110, -107, -104, -100, -96, -91, -86, -80, -73, -66, -58, -49, -40, -31, -21, -11, -1, 10, 20, 30, 39, 48, 57, 65, 72, 79, 85, 90, 95, 99, 103, 106, 109, 112, 114, 116, 117, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    _stage1_b2_block_block_1_act_Sigmoid_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _stage1_b2_block_block_1_act_Sigmoid_output_0_nl_params_data, _stage1_b2_block_block_1_act_Sigmoid_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_block_block_1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_1_dw_time_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_block_block_1_act_Sigmoid_output_0_layer, 158,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_stage1_b2_block_block_1_act_Sigmoid_output_0_chain,
  NULL, &_stage1_b2_block_block_1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = &_stage1_b2_block_block_1_act_Sigmoid_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_time_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_1_dw_freq_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_1_dw_time_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b2_block_block_1_dw_time_Conv_output_0_weights, &_stage1_b2_block_block_1_dw_time_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_1_dw_time_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_time_Conv_output_0_layer, 155,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_sssa8_ch,
  &_stage1_b2_block_block_1_dw_time_Conv_output_0_chain,
  NULL, &_stage1_b2_block_block_1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_freq_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_1_dw_freq_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b2_block_block_1_dw_freq_Conv_output_0_weights, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_1_dw_freq_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_block_block_1_dw_freq_Conv_output_0_layer, 152,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_sssa8_ch,
  &_stage1_b2_block_block_1_dw_freq_Conv_output_0_chain,
  NULL, &_stage1_b2_block_block_1_dw_time_Conv_output_0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_layer, 146,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_chain,
  NULL, &_stage1_b2_block_block_1_dw_freq_Conv_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_layer, 137,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_chain,
  NULL, &_stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_after_block_activation_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_layer, 137,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_chain,
  NULL, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_layer, AI_STATIC, 
  .groups = 4, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_after_block_activation_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_after_block_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_after_block_activation_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_after_block_activation_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_layer, 134,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b1_after_block_activation_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_chain,
  NULL, &_stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_after_block_activation_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b1_fuse_mode_Mul_1_output_0_output, &_stage1_b1_fuse_mode_Mul_3_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_after_block_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_after_block_activation_Relu_output_0_layer, 134,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b1_after_block_activation_Relu_output_0_chain,
  NULL, &_stage1_b1_after_block_activation_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b1_fuse_mode_Mul_2_output_0_output, &_stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_Mul_3_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_3_output_0_layer, 131,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b1_fuse_mode_Mul_3_output_0_chain,
  NULL, &_stage1_b1_after_block_activation_Relu_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_in_c_in_c_1_in_c_1_2_Relu_output_0_output, &_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_2_output_0_layer, 60,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b1_fuse_mode_Mul_2_output_0_chain,
  NULL, &_stage1_b1_fuse_mode_Mul_3_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b1_fuse_mode_Mul_output_0_output, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_Mul_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_1_output_0_layer, 125,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b1_fuse_mode_Mul_1_output_0_chain,
  NULL, &_stage1_b1_fuse_mode_Mul_2_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_layer, 126,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_chain,
  NULL, &_stage1_b1_fuse_mode_Mul_1_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Sub_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b1_fuse_mode_Constant_2_output_0_4D, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Sub_output_0_layer, 126,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_stage1_b1_fuse_mode_Sub_output_0_chain,
  NULL, &_stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_layer, 116,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_chain,
  NULL, &_stage1_b1_fuse_mode_Sub_output_0_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params_data[] = { -89, -89, -88, -88, -87, -87, -87, -86, -86, -85, -85, -85, -84, -84, -83, -83, -82, -82, -81, -81, -80, -80, -80, -79, -79, -78, -78, -77, -77, -76, -76, -75, -75, -74, -74, -73, -73, -72, -72, -71, -70, -70, -69, -69, -68, -68, -67, -66, -66, -65, -65, -64, -64, -63, -62, -62, -61, -61, -60, -59, -59, -58, -57, -57, -56, -55, -55, -54, -53, -53, -52, -51, -51, -50, -49, -49, -48, -47, -47, -46, -45, -44, -44, -43, -42, -41, -41, -40, -39, -38, -38, -37, -36, -35, -34, -34, -33, -32, -31, -30, -30, -29, -28, -27, -26, -26, -25, -24, -23, -22, -21, -20, -20, -19, -18, -17, -16, -15, -14, -13, -12, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127 };
AI_ARRAY_OBJ_DECLARE(
    _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params_data, _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_layer, 116,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_chain,
  NULL, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_layer, AI_STATIC, 
  .nl_params = &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_weights, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_layer, 113,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_chain,
  NULL, &_stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_layer, 104,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_chain,
  NULL, &_stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(17, 64), 
  .pool_stride = AI_SHAPE_2D_INIT(17, 64), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output0, &_in_c_in_c_1_in_c_1_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Add_output_0_layer, 98,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b1_fuse_mode_Add_output_0_chain,
  NULL, &_stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output0, &_stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_fuse_mode_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_fuse_mode_Mul_output_0_layer, 99,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b1_fuse_mode_Mul_output_0_chain,
  NULL, &_stage1_b1_fuse_mode_Add_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_layer, 92,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_chain,
  NULL, &_stage1_b1_fuse_mode_Mul_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_layer, 86,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_chain,
  NULL, &_stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_1_act_Mul_output_0_0_0__stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_layer, 86,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_chain,
  NULL, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_layer, AI_STATIC, 
  .groups = 4, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_block_block_1_act_Mul_output_0_0_0__stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_1_act_Mul_output_0_0_0__stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_block_block_1_act_Mul_output_0_0_0__stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_layer, 83,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b1_block_block_1_act_Mul_output_0_0_0__stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_chain,
  NULL, &_stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_block_block_1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stage1_b1_block_block_1_dw_time_Conv_output_0_output, &_stage1_b1_block_block_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_block_block_1_act_Mul_output_0_layer, 83,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_stage1_b1_block_block_1_act_Mul_output_0_chain,
  NULL, &_stage1_b1_block_block_1_act_Mul_output_0_0_0__stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 _stage1_b1_block_block_1_act_Sigmoid_output_0_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -126, -126, -125, -125, -124, -123, -122, -121, -120, -118, -116, -113, -110, -107, -102, -97, -91, -85, -77, -69, -59, -49, -37, -26, -13, -1, 12, 25, 36, 48, 58, 68, 76, 84, 90, 96, 101, 106, 109, 112, 115, 117, 119, 120, 121, 122, 123, 124, 124, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    _stage1_b1_block_block_1_act_Sigmoid_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _stage1_b1_block_block_1_act_Sigmoid_output_0_nl_params_data, _stage1_b1_block_block_1_act_Sigmoid_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_block_block_1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_1_dw_time_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_block_block_1_act_Sigmoid_output_0_layer, 80,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_stage1_b1_block_block_1_act_Sigmoid_output_0_chain,
  NULL, &_stage1_b1_block_block_1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = &_stage1_b1_block_block_1_act_Sigmoid_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_time_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_1_dw_time_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b1_block_block_1_dw_time_Conv_output_0_weights, &_stage1_b1_block_block_1_dw_time_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_1_dw_time_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_time_Conv_output_0_layer, 77,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_sssa8_ch,
  &_stage1_b1_block_block_1_dw_time_Conv_output_0_chain,
  NULL, &_stage1_b1_block_block_1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_freq_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_weights, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_block_block_1_dw_freq_Conv_output_0_layer, 74,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_sssa8_ch,
  &_stage1_b1_block_block_1_dw_freq_Conv_output_0_chain,
  NULL, &_stage1_b1_block_block_1_dw_time_Conv_output_0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_layer, 68,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_chain,
  NULL, &_stage1_b1_block_block_1_dw_freq_Conv_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_layer, 59,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_chain,
  NULL, &_stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_in_c_in_c_1_in_c_1_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_layer, 59,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_chain,
  NULL, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_layer, AI_STATIC, 
  .groups = 4, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_in_c_in_c_1_in_c_1_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_in_c_in_c_1_in_c_1_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_layer, 56,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_in_c_in_c_1_in_c_1_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_chain,
  NULL, &_stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_in_c_in_c_0_in_c_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_in_c_in_c_1_in_c_1_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_in_c_in_c_1_in_c_1_2_Relu_output_0_weights, &_in_c_in_c_1_in_c_1_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_in_c_in_c_1_in_c_1_2_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _in_c_in_c_1_in_c_1_2_Relu_output_0_layer, 56,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &_in_c_in_c_1_in_c_1_2_Relu_output_0_chain,
  NULL, &_in_c_in_c_1_in_c_1_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _in_c_in_c_0_in_c_0_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logmel_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_in_c_in_c_0_in_c_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_in_c_in_c_0_in_c_0_2_Relu_output_0_weights, &_in_c_in_c_0_in_c_0_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_in_c_in_c_0_in_c_0_2_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _in_c_in_c_0_in_c_0_2_Relu_output_0_layer, 53,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &_in_c_in_c_0_in_c_0_2_Relu_output_0_chain,
  NULL, &_in_c_in_c_1_in_c_1_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 43752, 1, 1),
    43752, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 319872, 1, 1),
    319872, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &logmel_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &_detection_head_detection_head_2_GlobalAveragePool_output_0_output),
  &_in_c_in_c_0_in_c_0_2_Relu_output_0_layer, 0x4c6b260f, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 43752, 1, 1),
      43752, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 319872, 1, 1),
      319872, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &logmel_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &_detection_head_detection_head_2_GlobalAveragePool_output_0_output),
  &_in_c_in_c_0_in_c_0_2_Relu_output_0_layer, 0x4c6b260f, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    logmel_output_array.data = AI_PTR(g_network_activations_map[0] + 285348);
    logmel_output_array.data_start = AI_PTR(g_network_activations_map[0] + 285348);
    _in_c_in_c_0_in_c_0_2_Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 285056);
    _in_c_in_c_0_in_c_0_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 285056);
    _in_c_in_c_0_in_c_0_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 251264);
    _in_c_in_c_0_in_c_0_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 251264);
    _in_c_in_c_1_in_c_1_2_Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 245920);
    _in_c_in_c_1_in_c_1_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 245920);
    _in_c_in_c_1_in_c_1_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 285056);
    _in_c_in_c_1_in_c_1_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 285056);
    _in_c_in_c_1_in_c_1_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 145792);
    _in_c_in_c_1_in_c_1_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 145792);
    _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 69632);
    _stage1_b1_block_block_0_proj_proj_1_Transpose_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 69632);
    _stage1_b1_block_block_1_dw_freq_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_block_block_1_dw_freq_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_block_block_1_dw_freq_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 139264);
    _stage1_b1_block_block_1_dw_freq_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 139264);
    _stage1_b1_block_block_1_dw_time_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_block_block_1_dw_time_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_block_block_1_dw_time_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1220);
    _stage1_b1_block_block_1_dw_time_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1220);
    _stage1_b1_block_block_1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 70852);
    _stage1_b1_block_block_1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 70852);
    _stage1_b1_block_block_1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 215424);
    _stage1_b1_block_block_1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 215424);
    _stage1_b1_block_block_1_act_Mul_output_0_0_0__stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 6528);
    _stage1_b1_block_block_1_act_Mul_output_0_0_0__stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6528);
    _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 4352);
    _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4352);
    _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 143616);
    _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 143616);
    _stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_block_block_2_proj_proj_1_Transpose_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_fuse_mode_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b1_fuse_mode_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b1_fuse_mode_Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 69632);
    _stage1_b1_fuse_mode_Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 69632);
    _stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 32);
    _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 32);
    _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 56);
    _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 32);
    _stage1_b1_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b1_fuse_mode_Sub_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32);
    _stage1_b1_fuse_mode_Sub_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 160);
    _stage1_b1_fuse_mode_Sub_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 160);
    _stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 32);
    _stage1_b1_fuse_mode_Sub_output_0_0_1__stage1_b1_fuse_mode_Mul_3_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32);
    _stage1_b1_fuse_mode_Mul_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 69632);
    _stage1_b1_fuse_mode_Mul_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 69632);
    _stage1_b1_fuse_mode_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 64);
    _stage1_b1_fuse_mode_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 64);
    _stage1_b1_fuse_mode_Mul_3_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 104448);
    _stage1_b1_fuse_mode_Mul_3_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 104448);
    _stage1_b1_after_block_activation_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_after_block_activation_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b1_after_block_activation_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 180608);
    _stage1_b1_after_block_activation_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 180608);
    _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 104448);
    _stage1_b2_block_block_0_proj_proj_1_Transpose_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 104448);
    _stage1_b2_block_block_1_dw_freq_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_block_block_1_dw_freq_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_block_block_1_dw_freq_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 174080);
    _stage1_b2_block_block_1_dw_freq_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 174080);
    _stage1_b2_block_block_1_dw_time_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_block_block_1_dw_time_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_block_block_1_dw_time_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 36036);
    _stage1_b2_block_block_1_dw_time_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 36036);
    _stage1_b2_block_block_1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 105668);
    _stage1_b2_block_block_1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 105668);
    _stage1_b2_block_block_1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 250240);
    _stage1_b2_block_block_1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 250240);
    _stage1_b2_block_block_1_act_Mul_output_0_0_0__stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 41344);
    _stage1_b2_block_block_1_act_Mul_output_0_0_0__stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 41344);
    _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 39168);
    _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 39168);
    _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 178432);
    _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 178432);
    _stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_block_block_2_proj_proj_1_Transpose_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_fuse_mode_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 69632);
    _stage1_b2_fuse_mode_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 69632);
    _stage1_b2_fuse_mode_Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 104448);
    _stage1_b2_fuse_mode_Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 104448);
    _stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 34848);
    _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 34848);
    _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34872);
    _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34872);
    _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34816);
    _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 34848);
    _stage1_b2_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b2_fuse_mode_Sub_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34848);
    _stage1_b2_fuse_mode_Sub_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34976);
    _stage1_b2_fuse_mode_Sub_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34976);
    _stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 34848);
    _stage1_b2_fuse_mode_Sub_output_0_0_1__stage1_b2_fuse_mode_Mul_3_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34848);
    _stage1_b2_fuse_mode_Mul_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 104448);
    _stage1_b2_fuse_mode_Mul_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 104448);
    _stage1_b2_fuse_mode_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34880);
    _stage1_b2_fuse_mode_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34880);
    _stage1_b2_fuse_mode_Mul_3_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b2_fuse_mode_Mul_3_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b2_after_block_activation_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 285056);
    _stage1_b2_after_block_activation_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 285056);
    _stage1_b2_after_block_activation_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 145792);
    _stage1_b2_after_block_activation_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 145792);
    _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 69632);
    _stage1_b3_block_block_0_proj_proj_1_Transpose_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 69632);
    _stage1_b3_block_block_1_dw_freq_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_block_block_1_dw_freq_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_block_block_1_dw_freq_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 139264);
    _stage1_b3_block_block_1_dw_freq_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 139264);
    _stage1_b3_block_block_1_dw_time_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_block_block_1_dw_time_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_block_block_1_dw_time_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1220);
    _stage1_b3_block_block_1_dw_time_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1220);
    _stage1_b3_block_block_1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 38084);
    _stage1_b3_block_block_1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 38084);
    _stage1_b3_block_block_1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 74948);
    _stage1_b3_block_block_1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 74948);
    _stage1_b3_block_block_1_act_Mul_output_0_0_0__stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 111812);
    _stage1_b3_block_block_1_act_Mul_output_0_0_0__stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 111812);
    _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 73728);
    _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 73728);
    _stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_block_block_2_proj_proj_1_Transpose_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_fuse_mode_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 18432);
    _stage1_b3_fuse_mode_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 18432);
    _stage1_b3_shortcut_shortcut_0_AveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 36864);
    _stage1_b3_shortcut_shortcut_0_AveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 36864);
    _stage1_b3_fuse_mode_Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 55296);
    _stage1_b3_fuse_mode_Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 55296);
    _stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 32);
    _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 32);
    _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 56);
    _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 32);
    _stage1_b3_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage1_b3_fuse_mode_Sub_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32);
    _stage1_b3_fuse_mode_Sub_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 160);
    _stage1_b3_fuse_mode_Sub_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 160);
    _stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 32);
    _stage1_b3_fuse_mode_Sub_output_0_0_1__stage1_b3_fuse_mode_Mul_3_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32);
    _stage1_b3_fuse_mode_Mul_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 55296);
    _stage1_b3_fuse_mode_Mul_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 55296);
    _stage1_b3_fuse_mode_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 64);
    _stage1_b3_fuse_mode_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 64);
    _stage1_b3_fuse_mode_Mul_3_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 18496);
    _stage1_b3_fuse_mode_Mul_3_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 18496);
    _stage1_b3_after_block_activation_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_after_block_activation_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _stage1_b3_after_block_activation_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 18432);
    _stage1_b3_after_block_activation_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 92160);
    _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 92160);
    _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 55296);
    _stage2_b4_block_block_0_proj_proj_1_Transpose_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 55296);
    _stage2_b4_block_block_1_dw_freq_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_block_block_1_dw_freq_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_block_block_1_dw_freq_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 20036);
    _stage2_b4_block_block_1_dw_freq_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 20036);
    _stage2_b4_block_block_1_dw_time_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_block_block_1_dw_time_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_block_block_1_dw_time_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 38468);
    _stage2_b4_block_block_1_dw_time_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 38468);
    _stage2_b4_block_block_1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_block_block_1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_block_block_1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 56900);
    _stage2_b4_block_block_1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56900);
    _stage2_b4_block_block_1_act_Mul_output_0_0_0__stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 75332);
    _stage2_b4_block_block_1_act_Mul_output_0_0_0__stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 75332);
    _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 149060);
    _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 149060);
    _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b4_block_block_2_proj_proj_1_Transpose_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b4_after_block_activation_Relu_output_0_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_after_block_activation_Relu_output_0_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b4_after_block_activation_Relu_output_0_0_conversion_0_0__stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b4_after_block_activation_Relu_output_0_0_conversion_0_0__stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 99072);
    _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 99072);
    _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_0_0__stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 69120);
    _stage2_b5_block_block_0_proj_proj_1_Transpose_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 69120);
    _stage2_b5_block_block_1_dw_freq_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_block_block_1_dw_freq_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_block_block_1_dw_freq_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 103680);
    _stage2_b5_block_block_1_dw_freq_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 103680);
    _stage2_b5_block_block_1_dw_time_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_block_block_1_dw_time_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_block_block_1_dw_time_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 36844);
    _stage2_b5_block_block_1_dw_time_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 36844);
    _stage2_b5_block_block_1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 71404);
    _stage2_b5_block_block_1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 71404);
    _stage2_b5_block_block_1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 105964);
    _stage2_b5_block_block_1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 105964);
    _stage2_b5_block_block_1_act_Mul_output_0_0_0__stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 140524);
    _stage2_b5_block_block_1_act_Mul_output_0_0_0__stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 140524);
    _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 99072);
    _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_0_0__stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 99072);
    _stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_block_block_2_proj_proj_1_Transpose_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_fuse_mode_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 50688);
    _stage2_b5_fuse_mode_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 50688);
    _stage2_b5_fuse_mode_Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 66816);
    _stage2_b5_fuse_mode_Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 66816);
    _stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_fuse_mode_channel_att_avg_pool_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 34616);
    _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 34616);
    _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34640);
    _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34640);
    _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 34616);
    _stage2_b5_fuse_mode_channel_att_sigmoid_Sigmoid_output_0_0_1__stage2_b5_fuse_mode_Sub_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34616);
    _stage2_b5_fuse_mode_Sub_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34840);
    _stage2_b5_fuse_mode_Sub_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34840);
    _stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 34616);
    _stage2_b5_fuse_mode_Sub_output_0_0_1__stage2_b5_fuse_mode_Mul_3_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34616);
    _stage2_b5_fuse_mode_Mul_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 66816);
    _stage2_b5_fuse_mode_Mul_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 66816);
    _stage2_b5_fuse_mode_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34672);
    _stage2_b5_fuse_mode_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34672);
    _stage2_b5_fuse_mode_Mul_3_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b5_fuse_mode_Mul_3_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 18432);
    _stage2_b5_after_block_activation_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 34560);
    _stage2_b5_after_block_activation_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34560);
    _aff_fusion_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 18432);
    _aff_fusion_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 18432);
    _MaxPool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 50688);
    _MaxPool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 50688);
    _Mul_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 59904);
    _Mul_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 59904);
    _AveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 50688);
    _AveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 50688);
    _Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 9216);
    _Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9216);
    _channel_adjust_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _channel_adjust_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _channel_adjust_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 50688);
    _channel_adjust_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 50688);
    _aff_fusion_Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _aff_fusion_Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 16128);
    _aff_fusion_channel_att_avg_pool_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16128);
    _aff_fusion_channel_att_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _aff_fusion_channel_att_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _aff_fusion_channel_att_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 24);
    _aff_fusion_channel_att_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24);
    _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 80);
    _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 80);
    _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 136);
    _aff_fusion_channel_att_sigmoid_Sigmoid_output_0_0_1__aff_fusion_Sub_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 136);
    _aff_fusion_Sub_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 360);
    _aff_fusion_Sub_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 360);
    _aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _aff_fusion_Sub_output_0_0_1__aff_fusion_Mul_3_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _aff_fusion_Mul_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 136);
    _aff_fusion_Mul_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 136);
    _aff_fusion_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 16264);
    _aff_fusion_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16264);
    _aff_fusion_Mul_3_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 32392);
    _aff_fusion_Mul_3_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32392);
    _aff_fusion_Add_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 16264);
    _aff_fusion_Add_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16264);
    _detection_head_detection_head_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _detection_head_detection_head_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _detection_head_detection_head_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 324);
    _detection_head_detection_head_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 324);
    _detection_head_detection_head_2_GlobalAveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _detection_head_detection_head_2_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _stage1_b1_fuse_mode_Constant_2_output_0_4D_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b1_fuse_mode_Constant_2_output_0_4D_array.data = AI_PTR(g_network_weights_map[0] + 0);
    _stage1_b1_fuse_mode_Constant_2_output_0_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    _stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 4);
    _stage1_b1_fuse_mode_Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 4);
    _Sub_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _Sub_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 8);
    _Sub_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 8);
    _Sigmoid_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _Sigmoid_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 12);
    _Sigmoid_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 12);
    _in_c_in_c_0_in_c_0_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _in_c_in_c_0_in_c_0_2_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 16);
    _in_c_in_c_0_in_c_0_2_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 16);
    _in_c_in_c_0_in_c_0_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _in_c_in_c_0_in_c_0_2_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 88);
    _in_c_in_c_0_in_c_0_2_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 88);
    _in_c_in_c_1_in_c_1_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _in_c_in_c_1_in_c_1_2_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 120);
    _in_c_in_c_1_in_c_1_2_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 120);
    _in_c_in_c_1_in_c_1_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _in_c_in_c_1_in_c_1_2_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 2424);
    _in_c_in_c_1_in_c_1_2_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2424);
    _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 2552);
    _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2552);
    _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 4600);
    _stage1_b1_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4600);
    _stage1_b1_block_block_1_dw_freq_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b1_block_block_1_dw_freq_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 4856);
    _stage1_b1_block_block_1_dw_freq_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 4856);
    _stage1_b1_block_block_1_dw_freq_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b1_block_block_1_dw_freq_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5048);
    _stage1_b1_block_block_1_dw_freq_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5048);
    _stage1_b1_block_block_1_dw_time_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b1_block_block_1_dw_time_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5304);
    _stage1_b1_block_block_1_dw_time_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5304);
    _stage1_b1_block_block_1_dw_time_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b1_block_block_1_dw_time_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5496);
    _stage1_b1_block_block_1_dw_time_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5496);
    _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5752);
    _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5752);
    _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 7800);
    _stage1_b1_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 7800);
    _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 7928);
    _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 7928);
    _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 7932);
    _stage1_b1_fuse_mode_channel_att_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 7932);
    _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 7936);
    _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 7936);
    _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 9984);
    _stage1_b2_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 9984);
    _stage1_b2_block_block_1_dw_freq_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b2_block_block_1_dw_freq_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 10240);
    _stage1_b2_block_block_1_dw_freq_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 10240);
    _stage1_b2_block_block_1_dw_time_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b2_block_block_1_dw_time_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 10432);
    _stage1_b2_block_block_1_dw_time_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 10432);
    _stage1_b2_block_block_1_dw_time_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b2_block_block_1_dw_time_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 10624);
    _stage1_b2_block_block_1_dw_time_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 10624);
    _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 10880);
    _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 10880);
    _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 12928);
    _stage1_b2_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 12928);
    _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 13056);
    _stage1_b2_fuse_mode_channel_att_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 13056);
    _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 13060);
    _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 13060);
    _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 15108);
    _stage1_b3_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 15108);
    _stage1_b3_block_block_1_dw_freq_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b3_block_block_1_dw_freq_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 15364);
    _stage1_b3_block_block_1_dw_freq_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 15364);
    _stage1_b3_block_block_1_dw_time_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b3_block_block_1_dw_time_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 15556);
    _stage1_b3_block_block_1_dw_time_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 15556);
    _stage1_b3_block_block_1_dw_time_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b3_block_block_1_dw_time_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 15748);
    _stage1_b3_block_block_1_dw_time_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 15748);
    _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 16004);
    _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 16004);
    _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 18052);
    _stage1_b3_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 18052);
    _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 18180);
    _stage1_b3_fuse_mode_channel_att_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 18180);
    _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 18184);
    _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 18184);
    _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 20232);
    _stage2_b4_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 20232);
    _stage2_b4_block_block_1_dw_freq_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b4_block_block_1_dw_freq_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 20488);
    _stage2_b4_block_block_1_dw_freq_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 20488);
    _stage2_b4_block_block_1_dw_time_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b4_block_block_1_dw_time_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 20808);
    _stage2_b4_block_block_1_dw_time_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 20808);
    _stage2_b4_block_block_1_dw_time_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b4_block_block_1_dw_time_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 21000);
    _stage2_b4_block_block_1_dw_time_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 21000);
    _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 21256);
    _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 21256);
    _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 24840);
    _stage2_b4_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 24840);
    _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 25064);
    _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 25064);
    _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 31784);
    _stage2_b5_block_block_0_proj_proj_0_proj_0_2_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 31784);
    _stage2_b5_block_block_1_dw_freq_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b5_block_block_1_dw_freq_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 32264);
    _stage2_b5_block_block_1_dw_freq_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 32264);
    _stage2_b5_block_block_1_dw_freq_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b5_block_block_1_dw_freq_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 32864);
    _stage2_b5_block_block_1_dw_freq_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 32864);
    _stage2_b5_block_block_1_dw_time_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b5_block_block_1_dw_time_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 33344);
    _stage2_b5_block_block_1_dw_time_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 33344);
    _stage2_b5_block_block_1_dw_time_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b5_block_block_1_dw_time_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 33704);
    _stage2_b5_block_block_1_dw_time_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 33704);
    _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 34184);
    _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 34184);
    _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 40904);
    _stage2_b5_block_block_2_proj_proj_0_proj_0_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 40904);
    _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 41128);
    _stage2_b5_fuse_mode_channel_att_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 41128);
    _channel_adjust_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _channel_adjust_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 41132);
    _channel_adjust_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 41132);
    _channel_adjust_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _channel_adjust_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 42924);
    _channel_adjust_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 42924);
    _aff_fusion_channel_att_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _aff_fusion_channel_att_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 43148);
    _aff_fusion_channel_att_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 43148);
    _detection_head_detection_head_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _detection_head_detection_head_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 43152);
    _detection_head_detection_head_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 43152);
    _detection_head_detection_head_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _detection_head_detection_head_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 43712);
    _detection_head_detection_head_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 43712);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 12956654,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x4c6b260f,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 12956654,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x4c6b260f,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

